# 导入黑白棋文件
from Reversi.AIPlayer import AIPlayer
from Reversi.HumanPlayer import HumanPlayer
from game import Game

# 人类玩家黑棋初始化
black_player = HumanPlayer("X")

# AI 玩家 白棋初始化
white_player = AIPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()
