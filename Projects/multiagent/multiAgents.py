# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        newFood = newFood.asList()#list
        ghostPos = []
        for G in newGhostStates:
            ghostPos_ = G.getPosition()[0], G. getPosition()[1]
            ghostPos.append(ghostPos_)
        #ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = newScaredTimes[0] > 0
        # if not new ScaredTimes new state is ghost: return lowest value
        if not scared and (newPos in ghostPos):
            return -2

        if scared and (newPos in ghostPos):
            return 1
        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = [util.manhattanDistance(fDist, newPos) for fDist in newFood]
        closestGhostDist = [util.manhattanDistance(gDist, newPos) for gDist in ghostPos]

        fd = min(closestFoodDist)

        gd = min(closestGhostDist)

        return 1 / fd - 1 / gd
        
        

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        pecmanAction = Directions.STOP
        pecmanValue = -100000

        for action in gameState.getLegalActions(0):
            v = self.getValue(gameState.generateSuccessor(0, action), 0, 1)
            if v > pecmanValue:
                pecmanValue = v
                pecmanAction = action
        return pecmanAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxAgent(gameState, currentDepth)
        else:
            return self.minAgent(gameState, currentDepth, agentIndex)
    
    def maxAgent(self, gameState, currentDepth):
        value = -100000
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            v = self.getValue(successorState, currentDepth, 1)
            value = max(v, value)
        return value
    
    def minAgent(self, gameState, currentDepth, agentIndex):
        value = 100000
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                value = min(value, self.getValue(gameState.generateSuccessor(agentIndex, action), 
                                                currentDepth + 1, 0))
            else:
                value = min(value, self.getValue(gameState.generateSuccessor(agentIndex, action),
                                                currentDepth, agentIndex + 1 ))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxAgent(gameState, 0, -100000, 100000)[1]
    
        
    def maxAgent(self, gameState, currentDepth, a, b):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState), None
        
        maxValue = -100000
        bestAction = None
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            value = self.minAgent(successorState, currentDepth, 1, a, b)[0]
            if value > maxValue:
                maxValue = value
                bestAction = action
            if value > b:
                return value, action
            a = max(a, value)
        return maxValue, bestAction
    
    def minAgent(self, gameState, currentDepth, agentIndex, a, b):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState), None
        
        minValue = 100000
        bestAction = None
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                value = self.maxAgent(successorState, currentDepth + 1, a, b)[0]
            else:
                value = self.minAgent(successorState, currentDepth, agentIndex + 1, a, b)[0]
            
            if value < minValue:
                minValue = value
                bestAction = action
            if value < a:
                return value, action
            b = min(b, value)
        return minValue, bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pecmanAction = Directions.STOP
        pecmanValue = -100000

        for action in gameState.getLegalActions(0):
            v = self.getValue(gameState.generateSuccessor(0, action), 0, 1)
            if v > pecmanValue:
                pecmanValue = v
                pecmanAction = action
        return pecmanAction

    def getValue(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxAgent(gameState, currentDepth)
        else:
            return self.expectAgent(gameState, currentDepth, agentIndex)
    
    def maxAgent(self, gameState, currentDepth):
        value = -100000
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            v = self.getValue(successorState, currentDepth, 1)
            value = max(v, value)
        return value
    
    def expectAgent(self, gameState, currentDepth, agentIndex):
        totalValue = 0
        count = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                totalValue += self.getValue(gameState.generateSuccessor(agentIndex, action), 
                                                currentDepth + 1, 0)
            else:
                totalValue += self.getValue(gameState.generateSuccessor(agentIndex, action),
                                                currentDepth, agentIndex + 1 )
            count += 1
        return totalValue / count
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    if len(food) > 0:
        nearestFood = min([manhattanDistance(pos, f) for f in food])
        foodHeuristic = 1 / nearestFood
    else:
        foodHeuristic = 0

    """
    ghostPos = []
    for G in ghostStates:
        ghostPos_ = G.getPosition()[0], G. getPosition()[1]
        ghostPos.append(ghostPos_)
    nearestGhost = min([manhattanDistance(pos, g) for g in ghostPos])
    scared = scaredTimes[0] > 0
    if scared:
        ghostHeuristic = nearestGhost * 0.1
    else:
        ghostHeuristic =  1 / float(nearestGhost + 1)
    
    return currentGameState.getScore() + foodHeuristic + ghostHeuristic
    util.raiseNotDefined()
    """
    return currentGameState.getScore() + foodHeuristic

# Abbreviation
better = betterEvaluationFunction
