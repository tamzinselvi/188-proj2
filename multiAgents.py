# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from search import mazeDistance
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self, *args):
      Agent.__init__(self, *args)
      self.lastStop = 1

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        move = legalMoves[chosenIndex]
        if move == Directions.STOP:
          self.lastStop = 1
        else:
          self.lastStop = 1

        return move

    def distance(self, pos1, pos2):
      try:
        return mazeDistance(pos1, pos2, self.state)
      except:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def isFacingMe(self, my, other, direction):
      if my[0] - other[0] < 0 and direction == 'East':
        return True
      if my[0] - other[0] > 0 and direction == 'West':
        return True
      if my[1] - other[1] < 0 and direction == 'South':
        return True
      if my[1] - other[1] > 0 and direction == 'North':
        return True
      return False

    def evaluationFunction(self, currentGameState, action):
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
        self.state = successorGameState
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        adversarys = filter(lambda ghostState: ghostState.scaredTimer <= 0, newGhostStates)
        adversaryDistances = [ self.distance(newPos, ghost.getPosition()) * ( 2 if self.isFacingMe(newPos, ghost.getPosition(), ghost.getDirection()) else 1 ) for ghost in adversarys]
        adversaryMetric = reduce(lambda x,y: x + 1./(y+.01), adversaryDistances, 0)

        scared = filter(lambda ghostState: ghostState.scaredTimer > 0, newGhostStates)
        scaredDistances = [ self.distance(newPos, ghost.getPosition()) for ghost in scared]
        scaredMetric = reduce(lambda x,y: x + 1./(y+.01), scaredDistances, 0)

        gridCords = [(x,y) for x in range(newFood.width) for y in range(newFood.height)]
        foodPos = filter( lambda pos: newFood[pos[0]][pos[1]], gridCords)
        oldFoodDistances = [ self.distance(currentGameState.getPacmanPosition(), pos) * ( .8 if self.isFacingMe(pos, oldPos, action) else 1 ) for pos in foodPos]
        newFoodDistances = [ self.distance(newPos, pos) * ( .8 if self.isFacingMe(pos, oldPos, action) else 1 )  for pos in foodPos]
        deltaFoodDistances = [ oldFoodDistances[i] - newFoodDistances[i] for i in range(len(newFoodDistances))]
        #foodMetric = reduce( lambda x,y: x + 1./(y+.1), newFoodDistances, 0)
        deltaFoodMetric = 100 if len(deltaFoodDistances) == 0 else reduce( lambda x,y: x+y, deltaFoodDistances, 0) / len(deltaFoodDistances)
       
        minFood = [index for index in range(len(newFoodDistances)) if newFoodDistances[index] == min(newFoodDistances)]
        distanceFoodMetric = 100 if len(newFoodDistances) == 0 else 1./min(newFoodDistances)
        deltaFoodMetric =  100 if len(newFoodDistances) == 0 else deltaFoodDistances[minFood[0]]

        #import pdb; pdb.set_trace()
        multiplier = 1 if action == Directions.STOP else self.lastStop
        return multiplier * (.3*successorGameState.getScore() +  .4*distanceFoodMetric + .1*deltaFoodMetric + scaredMetric - .5*adversaryMetric)

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        ret = self.minimaxSearch(gameState, 0)
        return ret[1]

    def minimaxSearch(self, gameState, depth):
        currentAgent = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(currentAgent)
        actionScores = []
        for action in actions:
          nextGameState = gameState.generateSuccessor(currentAgent, action)
          isTerminal = nextGameState.isWin() or nextGameState.isLose()
          if isTerminal or ( (depth + 1) // gameState.getNumAgents() ) == self.depth:
            actionScores += [self.evaluationFunction(nextGameState)]
          else:
            actionScores += [self.minimaxSearch(nextGameState, depth + 1)[0]]
        bestActionScore = max(actionScores) if currentAgent == 0 else min(actionScores)
        bestActionIndex = actionScores.index(bestActionScore)
        return (bestActionScore, actions[bestActionIndex])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        ret = self.alphaBetaSearch(gameState, 0, -sys.maxint - 1, sys.maxint)
        return ret[1]

    def alphaBetaSearch(self, gameState, depth, alpha, beta):
        currentAgent = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(currentAgent)
        actionScores = []
        for action in actions:
          nextGameState = gameState.generateSuccessor(currentAgent, action)
          isTerminal = nextGameState.isWin() or nextGameState.isLose()
          if isTerminal or ( (depth + 1) // gameState.getNumAgents() ) == self.depth:
            actionScores += [self.evaluationFunction(nextGameState)]
          else:
            actionScores += [self.alphaBetaSearch(nextGameState, depth + 1, alpha, beta)[0]]
          bestActionScore = max(actionScores) if currentAgent == 0 else min(actionScores)
          bestActionIndex = actionScores.index(bestActionScore)
          ret = (bestActionScore, actions[bestActionIndex])
          if currentAgent == 0:
            if bestActionScore > beta:
              return ret
            alpha = max(alpha, bestActionScore)
          else:
            if bestActionScore < alpha:
              return ret
            beta = min(beta, bestActionScore)
        return ret

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        ret = self.expectimaxSearch(gameState, 0)
        return ret[1]

    def expectimaxSearch(self, gameState, depth):
        currentAgent = depth % gameState.getNumAgents()
        actions = gameState.getLegalActions(currentAgent)
        actionScores = []
        for action in actions:
          nextGameState = gameState.generateSuccessor(currentAgent, action)
          isTerminal = nextGameState.isWin() or nextGameState.isLose()
          if isTerminal or ( (depth + 1) // gameState.getNumAgents() ) == self.depth:
            actionScores += [self.evaluationFunction(nextGameState)]
          else:
            actionScores += [self.expectimaxSearch(nextGameState, depth + 1)[0]]
        bestActionScore = max(actionScores) if currentAgent == 0 else float(sum(actionScores))/len(actionScores)
        bestAction = actions[actionScores.index(bestActionScore)] if currentAgent == 0 else None
        return (bestActionScore, bestAction)
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

