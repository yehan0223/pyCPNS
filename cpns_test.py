import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as vtk_to_numpy
from principal_nested_spheres import PNS

import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

### CPNS info
nSamples = 57
nRows = 3
nCols = 8
nTotalAtoms = nRows * nCols
nEndAtoms = 2 * nRows + 2 * (nCols - 2)
nStdAtoms = nTotalAtoms - nEndAtoms
nTotalPositions = 3 * (nEndAtoms + nStdAtoms)
nTotalRadii = 3 * nEndAtoms + 2 * nStdAtoms
MAX_CPNS_MODES = 15

### Read .vtp files
upSpokes = []
downSpokes = []
crestSpokes = []
for i in range(nSamples):
    sRepDir = 'data/hippocampus' + str(i)

    meshReader0 = vtk.vtkXMLPolyDataReader()
    meshReader0.SetFileName(sRepDir + "/up.vtp")
    meshReader0.Update()
    upSpokes.append(meshReader0.GetOutput())

    meshReader1 = vtk.vtkXMLPolyDataReader()
    meshReader1.SetFileName(sRepDir + "/down.vtp")
    meshReader1.Update()
    downSpokes.append(meshReader1.GetOutput())

    meshReader2 = vtk.vtkXMLPolyDataReader()
    meshReader2.SetFileName(sRepDir + "/crest.vtp")
    meshReader2.Update()
    crestSpokes.append(meshReader2.GetOutput())

### CPNS: Step 1 : Deal with hub Positions (PDM)
position = np.zeros((nTotalAtoms, 3, nSamples))
for i in range(nSamples):
    for j in range(nTotalAtoms):
        position[j, :, i] = upSpokes[i].GetPoint(j)

meanOfEachPDM = np.mean(position, 0,  keepdims=True)
meanOfCombinedPDM = np.mean(meanOfEachPDM, 2, keepdims=False)
cposition = position - np.repeat(meanOfEachPDM, nTotalAtoms, 0)
sscarryPDM = np.sqrt(np.sum(np.sum(cposition ** 2, 0), 0))
sphmatPDM = np.zeros((nTotalPositions, nSamples))
spharrayPDM = np.zeros((nTotalAtoms, 3, nSamples))
print(meanOfCombinedPDM.shape)
for i in range(nSamples):
    spharrayPDM[:, :, i] = cposition[:, :, i] / sscarryPDM[i]
    sphmatPDM[:, i] = np.reshape(spharrayPDM[:, :, i], (nTotalAtoms * 3))

# Fit PNS to data
pnsModel = PNS(sphmatPDM, itype=2)
pnsModel.fit()
ZShape, PNSShape = pnsModel.output

sizePDM = np.zeros((1, nSamples))
sizePDM[0, :] = sscarryPDM
meanSizePDM = np.exp(np.mean(np.log(sizePDM), keepdims=1))
normalizedSizePDM = np.log(sizePDM / meanSizePDM)

# Invert the resmat
# invertPNS = PNS.inv(resmat, PNS_coords)
# print(sphmatPDM) original data
# print(invertPNS) reconstructed data

### CPNS: Step 2 : Deal with atom radii (log radii are Euclidean variables)
logR = np.zeros((nTotalRadii, nSamples))
for i in range(nSamples):
    upR = vtk_to_numpy(upSpokes[i].GetPointData().GetScalars('spokeLength'))
    downR = vtk_to_numpy(downSpokes[i].GetPointData().GetScalars('spokeLength'))
    crestR = vtk_to_numpy(crestSpokes[i].GetPointData().GetScalars('spokeLength'))
    logR[:, i] = np.log(np.concatenate((upR, downR, crestR)))

meanLogR = np.mean(logR, axis=1, keepdims=1)
meanRs = np.exp(meanLogR)
rScaleFactors = np.repeat(meanRs, nSamples, axis=1)

uScaleFactors = np.zeros((2 * nTotalRadii, nSamples))
for i in range(nTotalRadii):
    uScaleFactors[2*i, :] = rScaleFactors[i, :]
    uScaleFactors[2*i+1, :] = rScaleFactors[i, :]

RStar = logR - np.repeat(meanLogR, nSamples, axis=1)

### CPNS: Step 3 : Deal with spoke directions (direction analysis)
spokeDirections = np.zeros((3*nTotalRadii, nSamples))
for i in range(nSamples):
    upD = np.reshape(vtk_to_numpy(upSpokes[i].GetPointData().GetArray('spokeDirection')), -1)
    downD = np.reshape(vtk_to_numpy(downSpokes[i].GetPointData().GetArray('spokeDirection')), -1)
    crestD = np.reshape(vtk_to_numpy(crestSpokes[i].GetPointData().GetArray('spokeDirection')), -1)
    spokeDirections[:, i] = np.concatenate((upD, downD, crestD))

print(spokeDirections[:, 0])

ZSpoke = np.zeros((2*nTotalRadii, nSamples))
PNSSpoke = []
for i in range(nTotalRadii):
    pnsModel = PNS(spokeDirections[(3*i):(3*i+3), :], itype=9)
    pnsModel.fit()
    zD, pnsD = pnsModel.output
    invert_zD = PNS.inv(zD, pnsD)
    print(zD)
    print(invert_zD)
    PNSSpoke.append(pnsD)
    ZSpoke[(2*i):(2*i+2), :] = zD

### CPNS: Step 4 : Construct composite Z matrix and perform PCA
ZComp = np.concatenate((np.multiply(meanSizePDM, ZShape),
                        np.multiply(meanSizePDM, normalizedSizePDM),
                        np.multiply(rScaleFactors, RStar),
                        np.multiply(uScaleFactors, ZSpoke)), axis=0)
cpns_pca = PCA()
cpns_pca.fit(ZComp.T)
explained_variance_ratio = cpns_pca.explained_variance_ratio_
num_components_cpns = len(explained_variance_ratio)
cpns_cum_ratio = []
for i in range(1, num_components_cpns + 1):
    cpns_cum_ratio.append(np.sum(explained_variance_ratio[:i]))

# Plot the graph for PCA proportions
fig, ax = plt.subplots()
ax.bar(range(1, 2*MAX_CPNS_MODES+1), explained_variance_ratio[:2*MAX_CPNS_MODES])
ax.plot(np.arange(1, 2*MAX_CPNS_MODES+1), cpns_cum_ratio[:2*MAX_CPNS_MODES], 'r-x')
ax.set_title('CPNS results')
ax.set_ylabel('Explained variance ratio')
ax.set_xlabel('Number of Principal components')
ax.set_xlim(0, 30)
ax.set_ylim(0, 1)
plt.show()

### CPNS: Step 5: Transform mean back from E-comp (eucliden) space to S (SRep) space
components_cpns = cpns_pca.components_.T

# 1. Convert from EComp to S-space (should be wrapped in a function)
CPNSMeanScores = np.zeros((components_cpns.shape[0], 1))

# Construct hub positions
CPNSScores = CPNSMeanScores
meanSizeOfPDMs = meanSizePDM

cpnsdim = CPNSMeanScores.shape[0]
dimZShape = 0
if cpnsdim < nTotalPositions + 3 * nTotalRadii:
    dimZShape = cpnsdim - 3 * nTotalRadii
else:
    dimZShape = nTotalPositions
zShape = CPNSScores[0:dimZShape-1] / meanSizePDM
zSizePDM = CPNSScores[dimZShape-1]
sizePDMOverall = np.multiply(meanSizeOfPDMs, np.exp(zSizePDM / meanSizeOfPDMs))[0, 0]
XStar = PNS.inv(zShape, PNSShape)[:, 0]
X = np.add(sizePDMOverall * XStar, np.repeat(meanOfCombinedPDM, nTotalAtoms))

# Construct radii
zRStar = CPNSScores[dimZShape: dimZShape + nTotalRadii]
radii = np.multiply(meanRs, np.exp(np.divide(zRStar, meanRs)))

# Construct spokeDirs
zSpokes = CPNSScores[dimZShape + nTotalRadii: dimZShape + 3 * nTotalRadii]
uScaleFactors = np.zeros((2 * nTotalRadii, 1))
for i in range(nTotalRadii):
    uScaleFactors[2*i, :] = meanRs[i]
    uScaleFactors[2*i+1, :] = meanRs[i]
zSpokes = np.divide(zSpokes, uScaleFactors)
spokeDirs = np.zeros((3*nTotalRadii, 1))

for ns in range(nTotalRadii):
    spokeDir = PNS.inv(zSpokes[2*ns:(2*ns+2), :], PNSSpoke[ns])
    spokeDirs[3*ns:(3*ns+3), [0]] = spokeDir

# 2. Write out the mean S-Rep
upMean = vtk.vtkPolyData()
upMean.DeepCopy(upSpokes[0])
for i in range(nTotalAtoms):
    upMean.GetPoints().SetPoint(i, X[3*i:3*i+3])

downMean = vtk.vtkPolyData()
downMean.DeepCopy(downSpokes[0])
for i in range(nTotalAtoms):
    downMean.GetPoints().SetPoint(i, X[3*i:3*i+3])

crestMean = vtk.vtkPolyData()
crestMean.DeepCopy(crestSpokes[0])
print('Done')