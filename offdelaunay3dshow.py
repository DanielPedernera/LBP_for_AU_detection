import vtk
import numpy as np
import sys

def openOff(name):
	m =[]
	sw2 = 0
	f = open(name,'r')
	for line in f:
		if sw2 > 2:
			m.append(line.strip(" \n").split(" "))
		else:
			sw2 += 1
	mat = np.array(np.double(m))
	return mat

def applyDelaunay3D(mat, alpha):
	points = vtk.vtkPoints()
	i = 0
	for x, y ,z in mat:
		points.InsertPoint(i, x, y ,z)
		i+=1
	profile = vtk.vtkPolyData()
	profile.SetPoints(points)
	delny = vtk.vtkDelaunay3D()
	delny.SetInputData(profile)
	delny.SetAlpha(alpha)
	delny.Update()
	geom = vtk.vtkGeometryFilter()
	geom.SetInputData(delny.GetOutput())
	geom.Update()
	return geom

def applyDelaunay2D(mat):
	points = vtk.vtkPoints()
	i = 0
	for x, y ,z in mat:
		points.InsertPoint(i, x, y ,z)
		i+=1
	profile = vtk.vtkPolyData()
	profile.SetPoints(points)
	delny = vtk.vtkDelaunay2D()
	delny.SetInput(profile)
	delny.Update()
	return delny

def applyDecimate(mesh):
	deci = vtk.vtkDecimatePro()
	deci.SetInputConnection(mesh.GetOutputPort())
	deci.SetTargetReduction(0.9)
	deci.PreserveTopologyOn()
	deci.Update()
	return deci

def applyLapliacianSmooth(mesh, its):
	smoother = vtk.vtkSmoothPolyDataFilter()
	smoother.SetInputConnection(mesh.GetOutputPort())
	smoother.SetNumberOfIterations(its)
	smoother.SetRelaxationFactor(0.1)
	smoother.FeatureEdgeSmoothingOff()
	smoother.BoundarySmoothingOn()
	smoother.Update()
	normals = vtk.vtkPolyDataNormals()
	normals.SetInputConnection(smoother.GetOutputPort())
	normals.ComputePointNormalsOn()
	normals.ComputeCellNormalsOn()
	normals.Update()
	return normals

def saveMeshAsPly(mesh, name):
	channel = open(name, 'wb')
	channel.close()
	w = vtk.vtkPLYWriter()
	w.SetInputConnection(mesh.GetOutputPort())
	w.SetFileName(name)
	w.SetFileTypeToBinary()
	w.SetDataByteOrderToLittleEndian()
	w.SetColorModeToUniformCellColor()
	w.SetColor(255, 255, 255)
	w.Write()
	print name + 'saved ...'

def showMesh(mesh):
	mapper = vtk.vtkPolyDataMapper()
	mapper.SetInputConnection(mesh.GetOutputPort())
	triangulation = vtk.vtkActor()
	triangulation.SetMapper(mapper)
	triangulation.GetProperty().SetColor(1, 1, 1)
	ren = vtk.vtkRenderer()
	renWin = vtk.vtkRenderWindow()
	renWin.AddRenderer(ren)
	iren = vtk.vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)
	ren.AddActor(triangulation)
	ren.SetBackground(0, 0, 0)
	renWin.SetSize(250, 250)
	renWin.Render()
	cam1 = ren.GetActiveCamera()
	cam1.Zoom(1.5)
	iren.Initialize()
	renWin.Render()
	iren.Start()

if __name__=="__main__":
		filename = str(sys.argv[1])
		basepath = str(sys.argv[2])
		#alpha = 2.5
		its = 15
		with open(filename) as base:
			for subject_au in base:
				subject = subject_au.split(' ')[0]
				inname = basepath + subject + '.off'
				ouname = basepath + subject + '_d2d.ply'
				mat = openOff(inname)
				#mesh = applyDelaunay3D(mat, alpha)
				#deci = applyDecimate(mesh)
				#s_mesh = applyLapliacianSmooth(deci, its)
				#saveMeshAsPly(s_mesh, ouname)
				mesh = applyDelaunay2D(mat)
				s_mesh = applyLapliacianSmooth(mesh, its)
				saveMeshAsPly(s_mesh, ouname)
		'''subject = sys.argv[1]
		outname = sys.argv[2]
		mat = openOff(subject)
		alpha = 1.5
		its = 50
		mesh = applyDelaunay(mat, alpha)
		s_mesh = applyLapliacianSmooth(mesh, its)
		saveMeshAsPly(s_mesh, outname)
		showMesh(s_mesh)'''
		sys.exit(0)
