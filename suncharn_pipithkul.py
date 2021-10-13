# Player suncharn_pipithkul

import numpy as np
from datetime import datetime


class GameState(object):
    __slots__ = ['board', 'playerToMove', 'gameOver', 'movesRemaining', 'points']


# Global variables
boardWidth = 0  # Board dimensions
boardHeight = 0
timeLimit = 0.0  # Maximum thinking time (in seconds) for each move
victoryPoints = 0  # Number of points for the winner
moveLimit = 0  # Maximum number of moves

defenseWeight = .10  # weight on how much we want to defend
attackWeight = .30  # weight on how much we want to attack
appleValue = 900
horseValue = 20

# If exceeded, game is a tie with victoryPoints being split between players.
# Otherwise, number of remaining moves is added to winner's score.
startState = None  # Initial state, provided to the initPlayer function
assignedPlayer = 0  # 1 -> player MAX; -1 -> player MIN (in terms of the MiniMax algorithm)
startTime = 0  # Remember the time stamp when move computation started

# Local parameters for player's algorithm. Can be modified, deleted, or extended in any conceivable way
pointMultiplier = 10  # Muliplier for winner's points in getScore function
pieceValue = 30  # Score value of a single piece in getScore function
victoryScoreThresh = 1000  # An absolute score exceeds this value if and only if one player has won
minLookAhead = 2  # Initial search depth for iterative deepening
maxLookAhead = 20  # Maximum search depth

moveDistanceFromAnySpot = None  # map how many move from any spot on the board to any given spot on the board
# pieceSquareTable = None  # piece square table for max player
maxPieceSquareTable = np.array(
    [
        [0, 0, 5, 0, 5, 0],
        [0, 0, 0, 0, 0, 0],
        [5, 0, 0, 12, 0, 8],
        [0, 0, 12, 0, 13, 0],
        [5, 12, 0, 0, 20, 13],
        [0, 0, 13, 20, 0, 0],
        [0, 8, 0, 13, 0, 100],
    ]
)
minPieceSquareTable = np.flip(maxPieceSquareTable)
maxDangerSquare = [(2, 1), (1, 2)]
maxSemiDangerSquare = [(4, 2), (3, 3), (2, 4), (4, 0), (3, 1), (1, 3), (0, 4), (2, 0), (0, 2)]
minDangerSquare = [(5, 3), (4, 4)]
minSemiDangerSquare = [(4, 1), (3, 2), (2, 3), (5, 2), (3, 4), (6, 1), (2, 5), (6, 3), (4, 5)]


# Compute list of legal moves for a given GameState and the player moving next
def getMoveOptions(state):
    # direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]  # Possible (dx, dy) moves
    direction = [(1, 2), (2, 1), (-1, 2), (-2, 1), (-1, -2), (-2, -1), (1, -2), (2, -1)]  # Possible (dx, dy) moves, move right first
    if assignedPlayer == -1:  # min player, starts right side of the board
        direction = [(-1, -2), (-2, -1), (1, -2), (2, -1), (1, 2), (2, 1), (-1, 2), (-2, 1)]  # Possible (dx, dy) moves, move left first
    moves = []

    for xStart in range(boardHeight):  # Search board for player's pieces
        for yStart in range(boardWidth):
            if state.board[xStart, yStart] == state.playerToMove:  # Found a piece!
                for (dx, dy) in direction:  # Check all potential move vectors
                    (xEnd, yEnd) = (xStart + dx, yStart + dy)
                    if xEnd >= 0 and xEnd < boardHeight and yEnd >= 0 and yEnd < boardWidth and not (
                            state.board[xEnd, yEnd] in [state.playerToMove, 2 * state.playerToMove]):
                        moves.append((xStart, yStart, xEnd,
                                      yEnd))  # If square is empty or occupied by the opponent, then we have a legal move.
    return moves

# For a given GameState and move to be executed, return the GameState that results from the move
def makeMove(state, move):
    (xStart, yStart, xEnd, yEnd) = move
    newState = GameState()
    newState.board = np.copy(state.board)  # The new board configuration is a copy of the current one except that...
    newState.board[xStart, yStart] = 0  # ... we remove the moving piece from its start position...
    newState.board[xEnd, yEnd] = state.playerToMove  # ... and place it at the end position
    newState.playerToMove = -state.playerToMove  # After this move, it will be the opponent's turn
    newState.movesRemaining = state.movesRemaining - 1
    newState.gameOver = False
    newState.points = 0

    if state.board[xEnd, yEnd] == -2 * state.playerToMove or not (-state.playerToMove in newState.board):
        newState.gameOver = True  # If the opponent lost the apple or all horses, the game is over...
        newState.points = state.playerToMove * (
                    victoryPoints + newState.movesRemaining)  # ... and more remaining moves result in more points
    elif newState.movesRemaining == 0:  # Otherwise, if there are no more moves left, the game is drawn
        newState.gameOver = True

    return newState


# Return the evaluation score for a given GameState; higher score indicates a better situation for Player MAX.
# Knight_Rider's evaluation function is based on the number of remaining horses and their proximity to the
# opponent's apple (the latter factor is not too useful in its current form but at least motivates Knight_Rider
# to move horses toward the opponent's apple).
def getScore(state):
    score = 0

    # check if the state is a winning state
    if assignedPlayer == 1 and state.board[boardHeight - 1, boardWidth - 1] == 1:
        return 10000
    elif assignedPlayer == -1 and state.board[0, 0] == -1:
        return -10000

    # Search board for any pieces and count scores
    for row in range(boardHeight):
        for col in range(boardWidth):
            score += materialValue(state.board[row, col])
            if state.board[row, col] == 1:
                score += maxPieceSquareTable[row, col]
            elif state.board[row, col] == -1:
                score -= minPieceSquareTable[row, col]
    # score += attackScore(state, 1)
    # score -= attackScore(state, -1)
    # score += defendScore(state, 1)
    # score -= defendScore(state, -1)

    return score

def materialValue(piece):
    if piece == 0:  # nothing there
        return 0
    elif piece == 2:  # max player apple
        return appleValue
    elif piece == -2:  # min player apple
        return -appleValue
    elif piece == 1:  # max player horse
        return horseValue
    elif piece == -1:  # min player horse
        return -horseValue

def attackScore(state, player):
    score = 0
    if player == 1:  # max player
        for (row, col) in minDangerSquare:
            score += numAttackValue(state, player, row, col)
        # for (row, col) in minSemiDangerSquare:
        #     score += numAttackValue(state, player, row, col) * .10
    else:  # min player
        for (row, col) in maxDangerSquare:
            score += numAttackValue(state, player, row, col)
        # for (row, col) in maxSemiDangerSquare:
        #     score += numAttackValue(state, player, row, col) * .10
    return score

def defendScore(state, player):
    score = 0
    if player == 1:  # max player
        for (row, col) in maxDangerSquare:
            score += numDefenseValue(state, player, row, col)
        # for (row, col) in maxSemiDangerSquare:
        #     score += numDefenseValue(state, player, row, col) * .10
    else:  # min player
        for (row, col) in minDangerSquare:
            score += numDefenseValue(state, player, row, col)
        # for (row, col) in minSemiDangerSquare:
        #     score += numDefenseValue(state, player, row, col) * .10
    return score


def numAttackValue(state, player, row, col):
    attackValue = 0
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    xStart, yStart = row, col
    for (dx, dy) in direction:
        (xEnd, yEnd) = (xStart + dx, yStart + dy)
        if 0 <= xEnd < boardHeight and 0 <= yEnd < boardWidth and state.board[xEnd, yEnd] == -player:
            attackValue += 1
    return attackValue

def numDefenseValue(state, player, row, col):
    defenseValue = 0
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
    xStart, yStart = row, col
    for (dx, dy) in direction:
        (xEnd, yEnd) = (xStart + dx, yStart + dy)
        if 0 <= xEnd < boardHeight and 0 <= yEnd < boardWidth and state.board[xEnd, yEnd] == player:
            defenseValue += 1
    return defenseValue

# def pieceSquareValue(piece, row, col):
#     if piece == 1:
#         return maxPieceSquareTable[row][col]
#     elif piece == -1:
#         return minPieceSquareTable[row][col]
#     else:
#         return 0


# Check whether time limit has been reached
def timeOut():
    # return False  # no time out for debugging purposes
    duration = datetime.now() - startTime
    return duration.seconds + duration.microseconds * 1e-6 >= timeLimit


# Use the minimax algorithm to look ahead <depthRemaining> moves and return the resulting score
def lookAhead(state, depthRemaining, alpha, beta):
    if depthRemaining == 0 or state.gameOver:
        return getScore(state)

    if timeOut():
        return 0

    bestScore = -9e9 * state.playerToMove

    for move in getMoveOptions(state):
        projectedState = makeMove(state, move)  # Try out every possible move...
        score = lookAhead(projectedState, depthRemaining - 1, alpha, beta)  # ... and score the resulting state

        if (state.playerToMove == 1 and score > bestScore) or (state.playerToMove == -1 and score < bestScore):
            bestScore = score  # Update bestScore if we have a new highest/lowest score for MAX/MIN

        # alpha beta pruning
        if state.playerToMove == 1:
            alpha = max(alpha, bestScore)
        else:
            beta = min(beta, bestScore)

        if beta <= alpha:
            break

    return bestScore


# Set global variables and initialize any data structures that the player will need
def initPlayer(_startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer):
    global startState, timeLimit, victoryPoints, moveLimit, assignedPlayer, boardWidth, boardHeight
    startState, timeLimit, victoryPoints, moveLimit, assignedPlayer = _startState, _timeLimit, _victoryPoints, _moveLimit, _assignedPlayer
    # startState.board = np.transpose(startState.board)  # swap row and height to get the correct dimension
    (boardHeight, boardWidth) = startState.board.shape

    initMoveDistanceFromAnySpot(boardHeight, boardWidth)
    pass

def initMoveDistance(startX, startY, boardHeight, boardWidth):
    direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]  # Possible (dx, dy) moves

    # starting vars for the loop
    moveDistanceMatrix = np.full((boardHeight, boardWidth), fill_value=-1)
    moveDistanceMatrix[startX, startY] = 0  # start pos
    processingTile = [(startX, startY)]
    numMove = processingAmount = 1

    # iterate through the tiles we are considering
    while len(processingTile) > 0:
        tile = processingTile.pop(0)
        processingAmount -= 1

        # try all moves from this tile
        for (dx, dy) in direction:
            (xEnd, yEnd) = (tile[0] + dx, tile[1] + dy)
            if 0 <= xEnd < boardHeight and 0 <= yEnd < boardWidth and moveDistanceMatrix[xEnd, yEnd] == -1:
                moveDistanceMatrix[xEnd, yEnd] = numMove
                processingTile.append((xEnd, yEnd))

        # next iteration of moves
        if processingAmount <= 0:
            processingAmount = len(processingTile)
            numMove += 1

    return moveDistanceMatrix

def initMoveDistanceFromAnySpot(boardHeight, boardWidth):
    global movesDistanceMatrix, moveDistanceFromAnySpot
    moveDistanceFromAnySpot = [x[:] for x in [[0] * boardWidth] * boardHeight]  # make a 2d array of temporary all elems = 0

    # create move distance from any spot matrix
    for iBoard in range(len(moveDistanceFromAnySpot)):
        for jBoard in range(len(moveDistanceFromAnySpot[iBoard])):
            moveDistanceFromAnySpot[iBoard][jBoard] = initMoveDistance(iBoard, jBoard, boardHeight, boardWidth)
    pass

# Free up memory if player used huge data structures
def exitPlayer():
    return

# Compute the next move to be played; keep updating <favoredMove> until computation finished or time limit reached
def getMove(state):
    global startTime
    startTime = datetime.now()  # Remember computation start time
    moveList = getMoveOptions(state)  # Get the list of possible moves
    favoredMove = moveList[0]  # Just choose first move from the list for now, in case we run out of time
    favoredMoveScore = -9e9 * state.playerToMove  # Use this variable to remember the score for the favored move

    # Iterative deepening loop
    for lookAheadDepth in range(minLookAhead, maxLookAhead + 1):
        currBestMove = None  # Best move and score currently found during the current iteration (lookAheadDepth)
        currBestScore = -9e9 * state.playerToMove

        # Try every possible next move, evaluate it using Minimax, and pick the one with best score
        for move in moveList:
            projectedState = makeMove(state, move)
            score = lookAhead(projectedState,
                              lookAheadDepth - 1, alpha=-9e9, beta=9e9)  # Find score through MiniMax for current lookAheadDepth

            if (state.playerToMove == 1 and score > currBestScore) or (
                    state.playerToMove == -1 and score < currBestScore):
                currBestMove, currBestScore = move, score  # Found new best move during this iteration

            if timeOut():
                break

        if not timeOut():  # Pick the move from the last lookahead depth as new favorite, unless the lookahead was incomplete
            favoredMove, favoredMoveScore = currBestMove, currBestScore
            duration = datetime.now() - startTime
            # print('Thomas: Depth %d finished at %.4f s, favored move (%d,%d)->(%d,%d), score = %.2f'
            #       % (lookAheadDepth, duration.seconds + duration.microseconds * 1e-6,
            #          favoredMove[0], favoredMove[1], favoredMove[2], favoredMove[3], favoredMoveScore))
        else:
            # print('Thomas: Timeout!')
            pass

        if timeOut() or abs(
                favoredMoveScore) > victoryScoreThresh:  # Stop computation if timeout or certain victory/defeat predicted
            break

    return favoredMove
