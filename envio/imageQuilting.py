# Implementação do algoritmo Image Quilting feita por Felipe Oliveira Magno Neves
# 180016296@aluno.unb.br
# Universidade de Brasília, 2021/1

# Uso das funções definidas se encontra no final do arquivo

# Importando bibliotecas a serem usadas pelo algoritmo
import cv2 as cv
from enum import Enum
import numpy as np
import builtins
import collections

# Enum criado para melhorar organização do código, definindo vertical como índice 0 e 
# horizontal como índice 1, quando necessário
class Direction(Enum):
    VERTICAL = 0
    HORIZONTAL = 1

#Função helper que apenas mostra as imagens passadas na lista de imagens imgs e 
# as mostra em janelas com nomes passados em names. Caso necessário, é possível 
# escolher a escala das janelas a serem apresentadas usando scaling.
def show(names, imgs, scaling = 1):
    for i in range(len(names)):
        cv.imshow(names[i], cv.resize(imgs[i], (imgs[i].shape[1] * scaling, imgs[i].shape[0] * scaling)))
    cv.waitKey(0)
    cv.destroyAllWindows()

# Função helper similar a show, mas não é passada lista de nomes para as janelas
def fShow(imgs, scaling = 1):
    for i in range(len(imgs)):
        if imgs[i] is not None:
            cv.imshow("image " + str(i), cv.resize(imgs[i], (imgs[i].shape[1] * scaling, imgs[i].shape[0] * scaling)))
    cv.waitKey(0)
    cv.destroyAllWindows()

#Função helper que, dadas duas imagens, as corta para o menor retângulo que pode ser 
# ocupada por ambas.
#Por exemplo, se é passada uma imagem 10x10 e uma imagem 11x9, retorna as duas imagens 
# cortadas para o formato 10x9.
def cutToSameShape(img1, img2):
    if(img1.shape[0] != img2.shape[0]):
        h = np.minimum(img1.shape[0], img2.shape[0])
        img1 = img1[:h]
        img2 = img2[:h]
    if(img1.shape[1] != img2.shape[1]):
        w = np.minimum(img1.shape[1], img2.shape[1])
        img1 = img1[:,:w]
        img2 = img2[:,:w]
    return(img1, img2)

# Função que computa (e retorna) a superficie de erro para a área de sobreposição de 
# duas imagens (patches). 
# Assume que img1 está, sempre, acima ou à esquerda de img2. 
# BoundarySize é a espessura da área de sobreposição. 
# Caso direction seja vertical, entende-se que img1 está à esquerda de img2, e a borda 
# entre elas tem orientação vertical.
def computeErrorSurface(img1, img2, boundarySize: int, direction: Direction, verbose = False):
    border1 = border2 = None
    if direction == Direction.HORIZONTAL:
        border1 = img1[-boundarySize-1:-1,:]
        border2 = img2[0:boundarySize,:]
    else:
        border1 = img1[:,-boundarySize-1:-1]
        border2 = img2[:,0:boundarySize]

    border1,border2 = cutToSameShape(border1, border2)

    if verbose:
        fShow([border1,border2], 3)

    diff = np.subtract(border1.astype(np.int),border2.astype(np.int))

    squareDiff = np.square(diff)

    ret = np.zeros((squareDiff.shape[0], squareDiff.shape[1]),dtype=np.uint32)

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            ret[i,j] = np.sum(squareDiff[i,j])

    ret = np.sqrt(ret)
    return cv.normalize(ret,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

#Funções helper do algoritmo adaptado de Dijkstra. Retornam os vizinhos acessíveis 
# a um pixel, dependendo se a borda (superfície de erro) sendo avaliada é 
# horizontal ou vetical
def verticalNeighbors(currentPos, errorBoundary):

    neighbors = []

    if(currentPos[0] != errorBoundary.shape[0]):
        neighbors.append((currentPos[0] - 1, currentPos[1]))
        if(currentPos[1] != 0):
            neighbors.append((currentPos[0] - 1, currentPos[1] - 1))
        if(currentPos[1] != errorBoundary.shape[1] - 1):
            neighbors.append((currentPos[0] - 1, currentPos[1] + 1))
    else:
        for i in range(errorBoundary.shape[1]):
            neighbors.append((currentPos[0] - 1, i))

    return neighbors

def horizontalNeighbors(currentPos, errorBoundary):

    neighbors = []

    if(currentPos[1] != errorBoundary.shape[1]):
        neighbors.append((currentPos[0], currentPos[1] - 1))
        if(currentPos[0] != 0):
            neighbors.append((currentPos[0] - 1, currentPos[1] - 1))
        if(currentPos[0] != errorBoundary.shape[0] - 1):
            neighbors.append((currentPos[0] + 1, currentPos[1] - 1))
    else:
        for i in range(errorBoundary.shape[0]):
            neighbors.append((i, currentPos[1] - 1))

    return neighbors

# Algoritmo adaptado de Dijkstra usado para computar o corte de borda de erro mínimo. 
# Retorna lista o registro da distância entre a borda da imagem e o corte que deve 
# ser feito.
# função fortemente inspirada pela implementação de Ben Alex Keen:
# https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
def minimumBoundaryCutDjikstras(errorBoundary, direction: Direction):
    dirIndex = direction.value
    invDirIndex = (direction.value + 1) % 2
    initial = [0,0]
    initial[dirIndex] = errorBoundary.shape[dirIndex]
    initial = tuple(initial)
    
    neighborFunction = verticalNeighbors

    if(direction == Direction.HORIZONTAL):
        neighborFunction = horizontalNeighbors

    shortestPaths = {initial: (None, 0)}
    currentNode = initial
    visited = set()
    
    while currentNode[dirIndex] != 0:
        visited.add(currentNode)
        destinations = neighborFunction(currentNode, errorBoundary)
        weightToCurrentNode = shortestPaths[currentNode][1]

        for nextNode in destinations:
            weight = errorBoundary[nextNode[0], nextNode[1]] + weightToCurrentNode
            if nextNode not in shortestPaths:
                shortestPaths[nextNode] = (currentNode, weight)
            else:
                currentShortestWeight = shortestPaths[nextNode][1]
                if currentShortestWeight > weight:
                    shortestPaths[nextNode] = (currentNode, weight)
        
        nextDestinations = {node: shortestPaths[node] for node in shortestPaths if node not in visited}
        currentNode = builtins.min(nextDestinations, key=lambda k: nextDestinations[k][1])
    
    path = []
    while currentNode is not None:
        path.append(currentNode[invDirIndex])
        nextNode = shortestPaths[currentNode][0]
        currentNode = nextNode

    return path[:-1]

# Função que aplica patch em imagem maior.
# é passada a imagem maior em 'fullImg', o patch em 'patch', e a posição (índice,
# não pixel) na qual o patch deve ser aplicado, em 'position'. 
# Esta posição é calculada a partir do tamanho do patch, que deve ser quadrado.
#'boundarySize' se refere à espessura da área de sobreposição/transição entre patches. 
# 'cutUp' e 'cutLeft' se referem aos cortes a serem aplicados no patch, que são 
# opcionais. Retorna 'fullImg' com o patch inserido na posição indicada.
def applyWithMask(fullImg, patch, position, boundarySize, cutUp = None, cutLeft = None, verbose=False):
    size = patch.shape[0] - boundarySize
    pixelPosition = [position[0]*size, position[1]*size]
    
    patchForPasting = cv.cvtColor(patch, cv.COLOR_RGB2RGBA)
    
    patchForPasting[:,:,3] = 255
    
    if(cutUp != None):
        np.clip(cutUp, 0, patchForPasting.shape[0] - 1)
        
        for i in range(patchForPasting.shape[1]):
            if i < len(cutUp):
                patchForPasting[:cutUp[i],i,3] = 0
            else:
                patchForPasting[:,i,3] = 0

    if(cutLeft != None):
        np.clip(cutLeft, 0, patchForPasting.shape[1] - 1)
        for i in range(patchForPasting.shape[0]):
            if i < len(cutLeft):
                patchForPasting[i,:cutLeft[i],3] = 0
            else:
                patchForPasting[i,:,3] = 0

    test = fullImg[pixelPosition[0]:pixelPosition[0] + patchForPasting.shape[0], pixelPosition[1]:pixelPosition[1] + patchForPasting.shape[1]].copy()

    patchForPasting, _ = cutToSameShape(patchForPasting, test)
    if verbose:
        fShow([patchForPasting[:, :, :3]  * (patchForPasting[:, :, 3:] / 255)], 3)

    fullImg[pixelPosition[0]:pixelPosition[0] + patchForPasting.shape[0], pixelPosition[1]:pixelPosition[1] + patchForPasting.shape[1]] = fullImg[pixelPosition[0]:pixelPosition[0] + patchForPasting.shape[0], pixelPosition[1]:pixelPosition[1] + patchForPasting.shape[1]] * (1 - patchForPasting[:, :, 3:] / 255) + patchForPasting[:, :, :3]  * (patchForPasting[:, :, 3:] / 255)
    return fullImg

# Computa o corte de borda de erro mínimo entre o patch atual (passado em 'currentImg') 
# e as áreas acima e à esquerda do mesmo, passadas em 'upImg' e 'leftImg' 
# respectivamente.
# Retorna lista com duas tuplas (uma para aborda vertical e outra para a horizontal).
# Cada tupla contém, no índice 0, o corte de erro mínimo, e no índice 1 a imagem da
# borda de erro.
def computeMinimumBoundaryCut(currentImg, boundarySize, upImg = None, leftImg = None):
    errorBoundaries = [(None, None), (None, None)]
    boundaryFlag = [False, False]

    if(upImg is not None):
        e = computeErrorSurface(upImg, currentImg, boundarySize, Direction.HORIZONTAL)
        errorBoundaries[Direction.HORIZONTAL.value] = e
        boundaryFlag[Direction.HORIZONTAL.value] = True

    if(leftImg is not None):
        e = computeErrorSurface(leftImg, currentImg, boundarySize, Direction.VERTICAL)
        errorBoundaries[Direction.VERTICAL.value] = e
        boundaryFlag[Direction.VERTICAL.value] = True

    if(boundaryFlag[0] and boundaryFlag[1]):
        minima = np.minimum(errorBoundaries[0][:boundarySize, :boundarySize], errorBoundaries[1][:boundarySize, :boundarySize])
        errorBoundaries[0][:boundarySize, :boundarySize] = errorBoundaries[1][:boundarySize, :boundarySize] = minima

    if(boundaryFlag[Direction.HORIZONTAL.value]):
        e = errorBoundaries[Direction.HORIZONTAL.value]
        minCut = minimumBoundaryCutDjikstras(e, Direction.HORIZONTAL)
        errorBoundaries[Direction.HORIZONTAL.value] = (minCut, e)

    if(boundaryFlag[Direction.VERTICAL.value]):
        e = errorBoundaries[Direction.VERTICAL.value]
        minCut = minimumBoundaryCutDjikstras(e, Direction.VERTICAL)
        errorBoundaries[Direction.VERTICAL.value] = (minCut, e)

    return errorBoundaries

# Função que retorna o erro entre as bordas do patch a ser aplicado e a 
# imagem original na posição de aplicação. retorna esse valor entre 0 e 1.
# maxValue é usado para limitar o erro ao intervalo (0,1)
def boundaryError(canvas, patch, position, borderSize, maxValue):
    mask = np.zeros(patch.shape, dtype=np.uint8)

    if position[0] != 0:
        mask[:borderSize,:] = 255

    if position[2] != 0:
        mask[:,:borderSize] = 255
          
    maskedPatch = cv.bitwise_and(patch, mask)
    canvasSlice = canvas[position[0]:position[1], position[2]:position[3]]

    maskedPatch, canvasSlice = cutToSameShape(maskedPatch, canvasSlice)
    return np.sum(np.divide(np.absolute(np.subtract(maskedPatch.astype(np.float),canvasSlice.astype(np.float))),maxValue))

# Função que calcula o erro entre os mapas de transferência na regiao com mesmo 
# tamano que o patch sendo aplicado. Usa maxValue para limitar esse erro 
# entre 0 e 1, e o retorna.
def correspondenceError(textureMap, targetMap, texturePosition, targetPosition, maxValue):
    texture = textureMap[texturePosition[0]:texturePosition[1], texturePosition[2]:texturePosition[3]]
    target = targetMap[targetPosition[0]:targetPosition[1], targetPosition[2]:targetPosition[3]]

    texture, target = cutToSameShape(texture, target)                                                                                                                         

    return np.sum(np.divide(np.absolute(np.subtract(target.astype(np.float),texture.astype(np.float))),maxValue))

# Função que, para a posição de aplicação passada, percorre toda a lista de patches 
# guardando os n menores erros de aplicação/sobreposição encontrados. Retorna o id
# de um patch escolhido aleatoriamente entre esses n menores erros.
def choosePatch(canvas, patches, positionId, cellSize, borderSize, textureMap, targetMap, patchSize, inputStep ,textureGridShape):
    global randomness
    global alpha

    bestPatch = collections.deque([(np.inf,0)], maxlen=randomness)
    targetPosition = [positionId[0]*cellSize,(positionId[0] + 1)*cellSize + borderSize, positionId[1]*cellSize,(positionId[1]+1)*cellSize + borderSize]
    maxValue = np.multiply(np.prod(patches[0].shape), 255)

    for i in range(len(patches)):
        row = int(i/textureGridShape[1])*inputStep
        col = (i%textureGridShape[1])*inputStep
        
        texturePosition = [row,row+patchSize, col,col+patchSize]
        error = alpha*boundaryError(canvas, patches[i], targetPosition, borderSize, maxValue)
        
        if(alpha != 1):
            error += (1-alpha)*correspondenceError(textureMap, targetMap, texturePosition, targetPosition,maxValue)

        if error < bestPatch[0][0]:
            bestPatch.appendleft((error, i))

    return bestPatch[np.random.randint(len(bestPatch))][1]

# Função que executa uma iteração do algoritmo Image Quilting, aplicando-a no 
# canvas caso o mesmo seja passado. Se não for fornecido um canvas inicial, 
# cria canvas preenchido com zeros. Retorna imgem preenchida com patches.
def quiltCanvas(patches, targetGridSize, cellSize, borderSize, textureMap, targetMap, patchSize, inputStep, textureGridShape, canvas=None):
    progress = 0
    progressIncrease = 100/np.prod(targetGridSize)
    
    if canvas is None:
        canvas = np.zeros((cellSize*targetGridSize[0] + borderSize, cellSize*targetGridSize[1] + borderSize, 3), dtype=np.uint8)
    else:
        canvas = canvas[:cellSize*targetGridSize[0] + borderSize, :cellSize*targetGridSize[1] + borderSize]
    
    for i in range(targetGridSize[0]):
        for j in range(targetGridSize[1]):
            try:
                print(str(int(progress)) + '%', end="\033[K\r")
                progress += progressIncrease
                
                upCut = leftCut = upImg = leftImg = curImg = None
                patchID = choosePatch(canvas, patches, (i,j), cellSize, borderSize, textureMap, targetMap, patchSize, inputStep, textureGridShape)

                curImg = patches[patchID]

                if i != 0:
                    upImg = canvas[(i-1)*cellSize:(i)*cellSize + borderSize, j*cellSize:(j+1)*cellSize + borderSize]
                if j != 0:
                    leftImg = canvas[i*cellSize:(i+1)*cellSize + borderSize, (j-1)*cellSize:j*cellSize + borderSize]

                temp = computeMinimumBoundaryCut(curImg, borderSize, upImg=upImg, leftImg=leftImg)
                upCut = temp[Direction.HORIZONTAL.value][0]
                leftCut = temp[Direction.VERTICAL.value][0]
                canvas = applyWithMask(canvas, curImg, (i,j), borderSize, cutLeft = leftCut, cutUp=upCut)
            except:
                pass
    return canvas

#---------------------aplicações do algoritmo se encontram abaixo----------------------#

# Variável que seleciona o tipo de execução a ser feita. Ela existe somente para 
# facilitar apresentação do algoritmo.

#execution = 'basic example'
#execution = 'texture synthesis'
execution = 'texture transfer'

# Código que gera imagens para demonstrar funcionamento básico do algoritmo
if execution == 'basic example':
    texture1 = cv.imread("./imagensParaTeste/input/noise1.jpg", cv.IMREAD_COLOR)
    texture2 = cv.imread("./imagensParaTeste/input/noise2.png", cv.IMREAD_COLOR)
    
    fShow([texture1, texture2], 3)

    errorBoundary = computeErrorSurface(texture2, texture1, 50, Direction.VERTICAL, True)

    fShow([errorBoundary], 3)

    cut = minimumBoundaryCutDjikstras(errorBoundary, Direction.VERTICAL)

    cutImage = cv.merge((errorBoundary,errorBoundary,errorBoundary))
    for i in range(len(cut)):
        cutImage[i,cut[i]] = [0,255,0]

    fShow([cutImage], 3)
    canvas = np.zeros((int(texture1.shape[0]), int(2*texture1.shape[1]) - 50, 3), dtype=np.uint8)

    canvas[:texture1.shape[0], :texture1.shape[1]] = texture1

    canvas = applyWithMask(canvas, texture2, (0,1), 50, cutLeft = cut, verbose=True)

    fShow([canvas], 3)

# Código que usa o algoritmo no modo de síntese de textura
if execution == 'texture synthesis':
    texturePath = "./imagensParaTeste/input/basket.png"
    textureSource = cv.imread(texturePath, cv.IMREAD_COLOR)
    textureSource = cv.resize(textureSource, (int(textureSource.shape[1]), int(textureSource.shape[0])))

    targetSource = cv.imread(texturePath, cv.IMREAD_COLOR)
    targetSource = cv.resize(targetSource, (int(targetSource.shape[1]*2), int(targetSource.shape[0]*2)))

    textureMap =    cv.GaussianBlur(cv.cvtColor(textureSource, cv.COLOR_BGR2GRAY), (9,9), 0.5)
    targetMap  =    cv.GaussianBlur(cv.cvtColor(targetSource, cv.COLOR_BGR2GRAY), (9,9), 0.5)

    canvas = None

    patchSize = 100
    inputStep = 10
    randomness = 2
    borderRatio = 4
    alpha = 1
    
    borderSize = int(patchSize/borderRatio)
    cellSize = patchSize - borderSize

    targetGridSize = (int(targetSource.shape[0]/cellSize),int(targetSource.shape[1]/cellSize))
    textureGridShape = (int((textureSource.shape[0] - patchSize)/inputStep),int((textureSource.shape[1] - patchSize)/inputStep))

    patches = []
    for i in range(textureGridShape[0]):
        for j in range(textureGridShape[1]):
            patches.append(textureSource[i*inputStep:i*inputStep + patchSize, j*inputStep:j*inputStep + patchSize].copy())
    
    canvas = quiltCanvas(patches, targetGridSize, cellSize, borderSize, textureMap, targetMap, patchSize, inputStep, textureGridShape, canvas=canvas)

    show(["texture", "output"], [textureSource, canvas], 2)

    cv.imwrite('./imagensParaTeste/output/texture_synthesis_' + '_'.join((str(randomness), str(alpha), str(patchSize), str(borderRatio))) + '.png', canvas)

# Código que usa o algoritmo no modo de transferência de textura
if execution == 'texture transfer':
    textureSource = cv.imread("./imagensParaTeste/input/textura.png", cv.IMREAD_COLOR)

    textureSource = cv.resize(textureSource, (int(textureSource.shape[1]/2), int(textureSource.shape[0]/2)))

    targetSource = cv.imread("./imagensParaTeste/input/alvo.png", cv.IMREAD_COLOR)
    targetSource = cv.resize(targetSource, (int(targetSource.shape[1]), int(targetSource.shape[0])))

    textureMap =    cv.GaussianBlur(cv.cvtColor(textureSource, cv.COLOR_BGR2GRAY), (11,11), 1)
    targetMap  =    cv.GaussianBlur(cv.cvtColor(targetSource, cv.COLOR_BGR2GRAY), (11,11), 1)

    textureMap = cv.equalizeHist(textureMap)
    targetMap = cv.equalizeHist(targetMap)

    fShow([textureMap, targetMap])

    canvas = None

    initialPatchSize = 50
    inputStep = 15
    randomness = 5
    borderRatio = 4
    N = 1

    patchSize = initialPatchSize*3/2
    for i in range(N+1):
        alpha =  0.5*(i/N) + 0.1
        patchSize = int(patchSize*2/3)
        borderSize = int(patchSize/borderRatio)
        cellSize = patchSize - borderSize

        if(borderSize == 0 or patchSize < inputStep):
            break

        targetGridSize = (int(targetSource.shape[0]/cellSize),int(targetSource.shape[1]/cellSize))
        textureGridShape = (int((textureSource.shape[0] - patchSize)/inputStep),int((textureSource.shape[1] - patchSize)/inputStep))

        patches = []
        for i in range(textureGridShape[0]):
            for j in range(textureGridShape[1]):
                patches.append(textureSource[i*inputStep:i*inputStep + patchSize, j*inputStep:j*inputStep + patchSize].copy())

        canvas = quiltCanvas(patches, targetGridSize, cellSize, borderSize, textureMap, targetMap, patchSize, inputStep, textureGridShape, canvas=canvas)

        show(["texture", "target", "output"], [textureSource, targetSource, canvas])

    cv.imwrite('./imagensParaTeste/output/texture_transfer_' + '_'.join((str(randomness), str(alpha), str(patchSize), str(borderRatio))) + '.png', canvas)