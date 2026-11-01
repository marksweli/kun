"""
Gmsh 网格生成：TBC 半周期单元 + off-peak 预制水平裂纹 (Case A, AR = 10/24)

单位: µm
  - 宽度 W = 12 µm (半波长, 左侧 x=0 为波峰/对称轴, 右侧 x=12 为波谷)
  - 基体 0 ~ 5000, BC 5000 ~ (界面), TGO 初始 1 µm, TC 顶面 5380
  - TC/TGO 界面  y_I(x) = 5130 + 5 cos(pi x / 12)   (峰谷高差 10 µm)
  - 预制裂纹: off-peak x_off = W/6 = 2 µm, 水平, 长 0.5 µm, 宽 0.06 µm, 方形裂尖,
              左端与 TC/TGO 界面相交 (只从 TC 中切除)
物理组: 1=TC, 2=TGO(初始), 3=BC, 4=Substrate
边界:   11=left(x=0), 12=right(x=W), 13=bottom(y=0), 14=top(y=H)
"""
import numpy as np
import gmsh

W = 12.0
H_SUB = 5000.0
Y_I = 5130.0          # TC/TGO 平均界面高度
AMP = 5.0             # 半幅值 (峰谷差 10 µm)
LAM = 24.0            # 波长
H_TOP = 5380.0
H_TGO0 = 1.0
X_OFF = W / 6.0       # off-peak 位置 (距波峰 1/6 总宽度)
CRACK_L = 0.5
CRACK_W = 0.06
H_FINE = 0.03         # 裂纹路径单元尺寸 = l_int / 2
H_BAND = 0.20         # TGO 生长带单元尺寸


def y_int(x):
    return Y_I + AMP * np.cos(2.0 * np.pi * np.asarray(x) / LAM)


Y_C = float(y_int(X_OFF))              # 裂纹中线高度
CRACK_TIP = (X_OFF + CRACK_L, Y_C)     # 右侧 (方形) 裂尖中心
FINE_BOX = (X_OFF - 1.0, X_OFF + 3.3, Y_C - 0.45, Y_C + 1.6)


def build(path="tbc_caseA.msh", verbose=False):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("tbc")
    occ = gmsh.model.occ

    xs = np.linspace(0.0, W, 97)

    def spline(offset):
        pts = [occ.addPoint(x, float(y_int(x)) - offset, 0) for x in xs]
        return occ.addSpline(pts), pts

    c0, p0 = spline(0.0)        # TC/TGO
    c1, p1 = spline(H_TGO0)     # TGO/BC

    pBL = occ.addPoint(0, H_SUB, 0); pBR = occ.addPoint(W, H_SUB, 0)
    pTL = occ.addPoint(0, H_TOP, 0); pTR = occ.addPoint(W, H_TOP, 0)

    # TC
    l1 = occ.addLine(p0[-1], pTR); l2 = occ.addLine(pTR, pTL); l3 = occ.addLine(pTL, p0[0])
    tc = occ.addPlaneSurface([occ.addCurveLoop([c0, l1, l2, l3])])
    # TGO initial
    l4 = occ.addLine(p1[-1], p0[-1]); l5 = occ.addLine(p0[0], p1[0])
    tgo = occ.addPlaneSurface([occ.addCurveLoop([c1, l4, -c0, l5])])
    # BC
    l6 = occ.addLine(pBR, p1[-1]); l7 = occ.addLine(p1[0], pBL); l8 = occ.addLine(pBL, pBR)
    bc = occ.addPlaneSurface([occ.addCurveLoop([l8, l6, -c1, l7])])
    # Substrate
    sub = occ.addRectangle(0, 0, 0, W, H_SUB)

    # 预制裂纹 (只从 TC 中切除)
    notch = occ.addRectangle(X_OFF - 0.3, Y_C - CRACK_W / 2, 0, 0.3 + CRACK_L, CRACK_W)
    tc_cut, _ = occ.cut([(2, tc)], [(2, notch)])
    occ.synchronize()
    out, omap = occ.fragment(tc_cut + [(2, tgo), (2, bc), (2, sub)], [])
    occ.synchronize()
    tc_t = [e[1] for e in omap[0]]
    tgo_t = [e[1] for e in omap[1]]
    bc_t = [e[1] for e in omap[2]]
    sub_t = [e[1] for e in omap[3]]
    gmsh.model.addPhysicalGroup(2, tc_t, 1, "TC")
    gmsh.model.addPhysicalGroup(2, tgo_t, 2, "TGO")
    gmsh.model.addPhysicalGroup(2, bc_t, 3, "BC")
    gmsh.model.addPhysicalGroup(2, sub_t, 4, "SUB")

    eps = 1e-6
    left, right, bot, top = [], [], [], []
    for (dim, tag) in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        if xmax < eps:
            left.append(tag)
        elif xmin > W - eps:
            right.append(tag)
        elif ymax < eps:
            bot.append(tag)
        elif ymin > H_TOP - eps:
            top.append(tag)
    gmsh.model.addPhysicalGroup(1, left, 11, "left")
    gmsh.model.addPhysicalGroup(1, right, 12, "right")
    gmsh.model.addPhysicalGroup(1, bot, 13, "bottom")
    gmsh.model.addPhysicalGroup(1, top, 14, "top")

    # ---- 尺寸场 ----
    fld = gmsh.model.mesh.field
    # (1) 背景: 离界面越远越粗, 最大 6 µm (宽度只有 12 µm)
    yI = f"({Y_I} + {AMP}*Cos(2*3.14159265358979*x/{LAM}))"
    f_bg = fld.add("MathEval")
    fld.setString(f_bg, "F", f"Min(6.0, Max({H_BAND}, 0.25*Abs(y - {yI} + 3.0) - 0.9))")
    # (2) TGO 生长带: 界面上方 1.5 µm ~ 下方 7.5 µm 内 0.2 µm
    f_band = fld.add("MathEval")
    fld.setString(f_band, "F", f"{H_BAND} + 10*Max(0, Abs(y - {yI} + 3.0) - 4.5)")
    # (3) 裂纹扩展区 (裂尖右上方) 0.03 µm
    x0, x1, y0, y1 = FINE_BOX
    f_box = fld.add("Box")
    fld.setNumber(f_box, "VIn", H_FINE); fld.setNumber(f_box, "VOut", 10.0)
    fld.setNumber(f_box, "XMin", x0); fld.setNumber(f_box, "XMax", x1)
    fld.setNumber(f_box, "YMin", y0); fld.setNumber(f_box, "YMax", y1)
    fld.setNumber(f_box, "Thickness", 0.8)
    f_min = fld.add("Min")
    fld.setNumbers(f_min, "FieldsList", [f_bg, f_band, f_box])
    fld.setAsBackgroundMesh(f_min)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    return gmsh.model


if __name__ == "__main__":
    build(verbose=True)
    n = len(gmsh.model.mesh.getElementsByType(2)[0])
    print("triangles:", n, " crack centre-line y_c =", Y_C)
    gmsh.finalize()
