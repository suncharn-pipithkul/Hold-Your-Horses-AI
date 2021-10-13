# Player Dark_Knight

import numpy as np
from datetime import datetime

class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']

# Global variables
boardWidth = 0            # Board dimensions 
boardHeight = 0          
timeLimit = 0.0           # Maximum thinking time (in seconds) for each move
victoryPoints = 0         # Number of points for the winner 
moveLimit = 0             # Maximum number of moves
                          # If exceeded, game is a tie with victoryPoints being split between players.
                          # Otherwise, number of remaining moves is added to winner's score.  
startState = None         # Initial state, provided to the initPlayer function
assignedPlayer = 0        # 1 -> player MAX; -1 -> player MIN (in terms of the MiniMax algorithm)
startTime = 0             # Remember the time stamp when move computation started

# Local parameters for player's algorithm. Can be modified, deleted, or extended in any conceivable way
name = 'Dark_Knight'
pointMultiplier = 10      # Muliplier for winner's points in getScore function
pieceValue = 20           # Score value of a single piece in getScore function
victoryScoreThresh = 1000 # An absolute score exceeds this value if and only if one player has won
minLookAhead = 3          # Initial search depth for iterative deepening
maxLookAhead = 20          # Maximum search depth 
evalCount = 0             # Count the total number of e(p) computations during each search cycle 
leafCount = 0             # Count the total number of leaves encountered during each search cycle 
appleDistance = None      # For exach square, contains distance (number of moves) to MAX or MIN's apple (index 0 or 1) 
posScore = None

direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]    # Possible (dx, dy) moves
    
# Compute list of legal moves for a given GameState and the player moving next 
def getMoveOptions(state):
    moves = []
    for xStart in range(boardWidth):                                    # Search board for player's pieces
        for yStart in range(boardHeight):
            if state.board[xStart, yStart] == state.playerToMove:       # Found a piece!
                for (dx, dy) in direction:                              # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight and not (state.board[xEnd, yEnd] in [state.playerToMove, 2 * state.playerToMove]):
                        moves.append((xStart, yStart, xEnd, yEnd))      # If square is empty or occupied by the opponent, then we have a legal move.
    return moves

# For a given GameState and move to be executed, return the GameState that results from the move
def makeMove(state, move):
    (xStart, yStart, xEnd, yEnd) = move
    newState = GameState()
    newState.board = np.copy(state.board)                   # The new board configuration is a copy of the current one except that...
    newState.board[xStart, yStart] = 0                      # ... we remove the moving piece from its start position...
    newState.board[xEnd, yEnd] = state.playerToMove         # ... and place it at the end position
    newState.playerToMove = -state.playerToMove             # After this move, it will be the opponent's turn
    newState.movesRemaining = state.movesRemaining - 1
    newState.gameOver = False
    newState.points = 0

    if state.board[xEnd, yEnd] == -2 * state.playerToMove or not (-state.playerToMove in newState.board):    
        newState.gameOver = True                            # If the opponent lost the apple or all horses, the game is over...
        newState.points = state.playerToMove * (victoryPoints + newState.movesRemaining)  # ... and more remaining moves result in more points
    elif newState.movesRemaining == 0:                      # Otherwise, if there are no more moves left, the game is drawn
        newState.gameOver = True
    
    return newState

# Return the evaluation score for a given GameState; higher score indicates a better situation for Player MAX.
# Knight_Rider's evaluation function is based on the number of remaining horses and their proximity to the 
# opponent's apple. 
def getScore(state):
    if state.gameOver:
        return pointMultiplier * state.points 

    score = 0
    playerToMoveIndex = (1 - state.playerToMove) // 2
    horseCount = [0, 0]
    fightingHorseCount = [0, 0]
    troubledHorseCount = [0, 0]
    protectCount = [0, 0]
    movesUntilWin = [10000, 10000]
    
    for x in range(boardWidth):                             # Search board for any pieces
        for y in range(boardHeight):                 
            if state.board[x, y] in [-1, 1]:
                playerCode = state.board[x, y]
                playerIndex = (1 - playerCode) // 2
                horseCount[playerIndex] += 1
                defendCount, attackCount = 0, 0
                score += playerCode * posScore[playerIndex, x, y]

                for (dx, dy) in direction:                              # Check all potential move vectors
                    (xEnd, yEnd) = (x + dx, y + dy)
                    if xEnd >= 0 and xEnd < boardWidth and yEnd >= 0 and yEnd < boardHeight:
                        if state.board[xEnd, yEnd] == playerCode:
                            defendCount += 1
                        elif state.board[xEnd, yEnd] == -playerCode:
                            attackCount += 1
                
                if attackCount > 0:
                    fightingHorseCount[playerIndex] += 1
                if attackCount > defendCount:
                    troubledHorseCount[playerIndex] += 1
                
                protectCount[playerIndex] += defendCount
                
                if appleDistance[1 - playerIndex, x, y] == 1:
                    if state.playerToMove == playerCode:        # 1 step away from opponent's apple and making next move -> win!
                        movesUntilWin[playerToMoveIndex] = 1
                    elif defendCount >= attackCount:
                        movesUntilWin[1 - playerToMoveIndex] = min(movesUntilWin[1 - playerToMoveIndex], 2 * attackCount + 2)
    
    score += pieceValue * (horseCount[0] - horseCount[1]) + 1 * (troubledHorseCount[1] - troubledHorseCount[0]) + 4.0 * (protectCount[0]/horseCount[0] - protectCount[1]/horseCount[1])
    if troubledHorseCount[playerToMoveIndex] < troubledHorseCount[1 - playerToMoveIndex] or (troubledHorseCount[1 - playerToMoveIndex] == 1 and fightingHorseCount[1 - playerToMoveIndex] == 1):  # Player can win a horse in the next move
        score += state.playerToMove * pieceValue
    
    if state.playerToMove == 1 and horseCount[1] == 1 and troubledHorseCount[1] == 1: # If MAX moves next and can kick out the opponent's last horse, game is over
        movesUntilWin[playerToMoveIndex] = 1 # -1 because it takes one more move to win
    elif state.playerToMove == -1 and horseCount[0] == 1 and troubledHorseCount[0] == 1 and state.movesRemaining > 0: # Same for MIN
        movesUntilWin[playerToMoveIndex] = 1

    if movesUntilWin[playerToMoveIndex] < 1000 and movesUntilWin[playerToMoveIndex] <= movesUntilWin[1 - playerToMoveIndex]:
        score = state.playerToMove * pointMultiplier * (victoryPoints + state.movesRemaining - movesUntilWin[playerToMoveIndex])
    elif movesUntilWin[1 - playerToMoveIndex] < movesUntilWin[playerToMoveIndex]:
        score = -state.playerToMove * pointMultiplier * (victoryPoints + state.movesRemaining - movesUntilWin[1 - playerToMoveIndex])
    
    return score

# Check whether time limit has been reached
def timeOut():
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit

# Use the minimax algorithm to look ahead <depthRemaining> moves and return the resulting score
def lookAhead(state, depthRemaining, alpha, beta):
    global leafCount
    if depthRemaining == 0 or state.gameOver:
        #leafCount += 1
        return getScore(state)

    if timeOut():
        return 0
        
    a, b = alpha, beta

    for move in getMoveOptions(state):
        projectedState = makeMove(state, move)
        score = lookAhead(projectedState, depthRemaining - 1, a, b)
        
        if state.playerToMove == 1 and score > a:
            a = score
            if a >= beta:
                break
        elif state.playerToMove == -1 and score < b:
            b = score
            if b <= alpha:
                break
    if state.playerToMove == 1:
        return a
    else:
        return b

# Use the minimax algorithm to look ahead <depthRemaining> moves and return the resulting score
def lookAheadWithPresort(state, depthRemaining, alpha, beta):
    global leafCount
    if depthRemaining == 0 or state.gameOver:
        #leafCount += 1
        return getScore(state)

    if timeOut():
        return 0
        
    a, b = alpha, beta
    scoreList = []
    projectedStateList = []
    moveList = getMoveOptions(state)

    for move in moveList:
        projectedState = GameState()
        projectedState = makeMove(state, move)
        scoreList.append(-state.playerToMove * getScore(projectedState))
        projectedStateList.append(projectedState)
    
    moveOrder = np.argsort(scoreList)
    if depthRemaining == 1:
        return -state.playerToMove * scoreList[moveOrder[0]]

    for moveIndex in moveOrder:
        if depthRemaining > 3:
            score = lookAheadWithPresort(projectedStateList[moveIndex], depthRemaining - 1, a, b)    # Find score through MiniMax for current lookAheadDepth
        else:
            score = lookAhead(projectedStateList[moveIndex], depthRemaining - 1, a, b)    # Find score through MiniMax for current lookAheadDepth

        if state.playerToMove == 1 and score > a:
            a = score
            if a >= beta:
                break
        elif state.playerToMove == -1 and score < b:
            b = score
            if b <= alpha:
                break
    if state.playerToMove == 1:
        return a
    else:
        return b

def setAppleDistance(player, xApple, yApple):
    global appleDistance
    numMissingSquares = boardWidth * boardHeight - 1
    appleDistance[player, xApple, yApple] = 0
    currLabel = -1

    while numMissingSquares > 0:
        currLabel += 1
        for x in range(boardWidth):
            for y in range(boardHeight):
                if appleDistance[player, x, y] == currLabel:
                    for (dx, dy) in direction:                              # Check all potential move vectors
                        (xNew, yNew) = (x + dx, y + dy)
                        if xNew >= 0 and xNew < boardWidth and yNew >= 0 and yNew < boardHeight and appleDistance[player, xNew, yNew] == -1:
                            appleDistance[player, xNew, yNew] = currLabel + 1
                            numMissingSquares -= 1

# Set global variables and initialize any data structures that the player will need
def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight, appleDistance, posScore
    
    startState, timeLimit, victoryPoints, moveLimit, assignedPlayer = _startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer 
    (boardWidth, boardHeight) = startState.board.shape
    
    appleDistance = -np.ones((2, boardWidth, boardHeight), dtype=int)
    posScore = np.zeros((2, boardWidth, boardHeight))
    defenseScore = [0, 2, 2, 0]
    attackScore = [0, 6, 3, 1]
    
    for x in range(boardWidth):
        for y in range(boardHeight):
            if startState.board[x, y] == 2:
                setAppleDistance(0, x, y)  # Fill in appleDistance values for MAX (in [0, :, :])
            elif startState.board[x, y] == -2:
                setAppleDistance(1, x, y)  # Fill in appleDistance values for MIN (in [1, :, :])

    for x in range(boardWidth):
        for y in range(boardHeight):
            edgeDistX = min(x, boardWidth - 1 - x)
            edgeDistY = min(y, boardHeight - 1 - y)
            centrality = min(2, edgeDistX) + min(2, edgeDistY)
            for pl in range(2):
                posScore[pl, x, y] = centrality
                if appleDistance[pl, x, y] < 4:
                    posScore[pl, x, y] += defenseScore[appleDistance[pl, x, y]]
                if appleDistance[1 - pl, x, y] < 4:
                    posScore[pl, x, y] += attackScore[appleDistance[1 - pl, x, y]]
                    
    defenseScore = np.zeros(2 * (boardWidth + boardHeight))
    
    
# Free up memory if player used huge data structures 
def exitPlayer():
    return

# Compute the next move to be played; keep updating <favoredMove> until computation finished or time limit reached
def getMove(state):
    global startTime, leafCount, evalCount
    
    #print('SCORE: ' + str(getScore(state)))
    #input("")

    startTime = datetime.now()                      # Remember computation start time
    moveList = getMoveOptions(state)                # Get the list of possible moves

    scoreList = []
    projectedStateList = []
    for move in moveList:
        projectedState = GameState()
        projectedState = makeMove(state, move)
        scoreList.append(-state.playerToMove * getScore(projectedState))
        projectedStateList.append(projectedState)

    moveOrder = np.argsort(scoreList)
    
    favoredMove = moveList[moveOrder[0]]            # Just choose first move from the list for now, in case we run out of time 
    favoredMoveScore = -9e9    # Use this variable to remember the score for the favored move  
    
    # Iterative deepening loop
    for lookAheadDepth in range(minLookAhead, maxLookAhead + 1):
        alpha = -9e9
        beta = 9e9
        currBestMove = None                         # Best move and absolute score during the current iteration (lookAheadDepth)                     
        currBestScore = alpha
        leafCount = 0
        evalCount = 0

        # Try every possible next move, evaluate it using Minimax, and pick the one with best score
        for moveIndex in moveOrder:
            move = moveList[moveIndex]                       
            score = lookAheadWithPresort(projectedStateList[moveIndex], lookAheadDepth - 1, alpha, beta)    # Find score through MiniMax for current lookAheadDepth
            
            if timeOut():
                break
            
            if state.playerToMove == 1 and score > alpha:
                alpha, currBestMove, currBestScore = score, move, score
            elif state.playerToMove == -1 and score < beta:
                beta, currBestMove, currBestScore = score, move, -score
        
        if timeOut():
            if favoredMoveScore < victoryScoreThresh and currBestScore > victoryScoreThresh:
                print(name + ': Timeout! Depth %d imcomplete but found winning move!'%(lookAheadDepth)) 
            elif favoredMoveScore < -victoryScoreThresh and currBestScore > -victoryScoreThresh:
                print(name + ': Timeout! Depth %d imcomplete but found defending move!'%(lookAheadDepth))
            else:
                print(name + ': Timeout! Depth %d imcomplete and disregarded!'%(lookAheadDepth))
                currBestMove, currBestScore = favoredMove, favoredMoveScore 
        
        favoredMove, favoredMoveScore = currBestMove, currBestScore
        
        duration = datetime.now() - startTime
        print(name + ': Depth %d finished at %.4f s, %d evals, %d leaves, favored move (%d,%d)->(%d,%d), score = %.2f'
            %(lookAheadDepth, duration.seconds + duration.microseconds * 1e-6, evalCount, leafCount, 
            favoredMove[0], favoredMove[1], favoredMove[2], favoredMove[3], favoredMoveScore))

        if timeOut() or abs(favoredMoveScore) > victoryScoreThresh:   # Stop computation if timeout or certain victory/defeat predicted
            break

    return favoredMove
