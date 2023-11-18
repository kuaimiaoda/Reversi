from Reversi.HumanPlayer import HumanPlayer
from board import Board
import datetime
import random
from math import log, sqrt
from time import time
from copy import deepcopy


class TreeNode():
    """
    蒙特卡洛树节点
    """

    def __init__(self, parent, color):
        self.parent = parent
        self.w = 0
        self.n = 0
        self.color = color
        self.child = dict()


def oppo(color):
    """
    交换棋手
    :return: 切换下一步落子棋手
    """

    if color == 'X':
        return 'O'
    return 'X'


class SilentGame(object):
    ''' 重构游戏类，模拟下棋过程中，不实时打印棋盘  '''

    def __init__(self, black_player, white_player, board=Board(), current_player=None):
        self.board = deepcopy(board)  # 棋盘
        # 定义棋盘上当前下棋棋手，先默认是 None
        self.current_player = current_player
        self.black_player = black_player  # 黑棋一方
        self.white_player = white_player  # 白棋一方
        self.black_player.color = "X"
        self.white_player.color = "O"

    def switch_player(self, black_player, white_player):
        """
        游戏过程中切换玩家
        :param black_player: 黑棋
        :param white_player: 白棋
        :return: 当前玩家
        """
        # 如果当前玩家是 None 或者 白棋一方 white_player，则返回 黑棋一方 black_player;
        if self.current_player is None:
            return black_player
        else:
            # 如果当前玩家是黑棋一方 black_player 则返回 白棋一方 white_player
            if self.current_player == self.black_player:
                return white_player
            else:
                return black_player

    def print_winner(self, winner):
        """
        打印赢家
        :param winner: [0,1,2] 分别代表黑棋获胜、白棋获胜、平局3种可能。
        :return:
        """
        print(['黑棋获胜!', '白棋获胜!', '平局'][winner])

    def force_loss(self, is_timeout=False, is_board=False, is_legal=False):
        """
         落子3个不合符规则和超时则结束游戏,修改棋盘也是输
        :param is_timeout: 时间是否超时，默认不超时
        :param is_board: 是否修改棋盘
        :param is_legal: 落子是否合法
        :return: 赢家（0,1）,棋子差 0
        """

        if self.current_player == self.black_player:
            win_color = '白棋 - O'
            loss_color = '黑棋 - X'
            winner = 1
        else:
            win_color = '黑棋 - X'
            loss_color = '白棋 - O'
            winner = 0

        if is_timeout:
            print('\n{} 思考超过 60s, {} 胜'.format(loss_color, win_color))
        if is_legal:
            print('\n{} 落子 3 次不符合规则,故 {} 胜'.format(loss_color, win_color))
        if is_board:
            print('\n{} 擅自改动棋盘判输,故 {} 胜'.format(loss_color, win_color))

        diff = 0

        return winner, diff

    def run(self):
        """
        运行游戏
        :return:
        """
        # 定义统计双方下棋时间
        total_time = {"X": 0, "O": 0}
        # 定义双方每一步下棋时间
        step_time = {"X": 0, "O": 0}
        # 初始化胜负结果和棋子差
        winner = None
        diff = -1

        # 游戏开始
        while True:
            # 切换当前玩家,如果当前玩家是 None 或者白棋 white_player，则返回黑棋 black_player;
            #  否则返回 white_player。
            self.current_player = self.switch_player(self.black_player, self.white_player)
            start_time = datetime.datetime.now()
            # 当前玩家对棋盘进行思考后，得到落子位置
            # 判断当前下棋方
            color = "X" if self.current_player == self.black_player else "O"
            # 获取当前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            # print("%s合法落子坐标列表："%color,legal_actions)
            if len(legal_actions) == 0:
                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束，双方都没有合法位置
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    break
                else:
                    # 另一方有合法位置,切换下棋方
                    continue

            action = self.current_player.get_move(self.board)

            if action is None:
                continue
            else:
                self.board._move(action, color)
                if self.game_over():
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    break

        return winner, diff

    def game_over(self):
        """
        判断游戏是否结束
        :return: True/False 游戏结束/游戏没有结束
        """

        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
        b_list = list(self.board.get_legal_actions('X'))
        w_list = list(self.board.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over


# 结合了多种策略，同时也结合了Mobility的特性，因为中间子的优先级较高，会提高自己的Mobility而限制对手的可走步数
class RoxannePlayer(object):
    ''' Roxanne 策略 详见 《Analysis of Monte Carlo Techniques in Othello》 '''
    ''' 提出者：Canosa, R. Roxanne canosa homepage. https://www.cs.rit.edu/~rlc/ '''

    def __init__(self, color):
        """
        Roxanne策略初始化
        :param roxanne_table: 从上到下依次按落子优先级排序
        :param color: 执棋方
        """

        self.roxanne_table = [
            ['A1', 'H1', 'A8', 'H8'],
            ['C3', 'F3', 'C6', 'F6'],
            ['C4', 'F4', 'C5', 'F5', 'D3', 'E3', 'D6', 'E6'],
            ['A3', 'H3', 'A6', 'H6', 'C1', 'F1', 'C8', 'F8'],
            ['A4', 'H4', 'A5', 'H5', 'D1', 'E1', 'D8', 'E8'],
            ['B3', 'G3', 'B6', 'G6', 'C2', 'F2', 'C7', 'F7'],
            ['B4', 'G4', 'B5', 'G5', 'D2', 'E2', 'D7', 'E7'],
            ['B2', 'G2', 'B7', 'G7'],
            ['A2', 'H2', 'A7', 'H7', 'B1', 'G1', 'B8', 'G8']
        ]
        self.color = color

    def roxanne_select(self, board):
        """
        采用Roxanne 策略选择落子策略
        :return: 落子策略
        """

        action_list = list(board.get_legal_actions(self.color))
        if len(action_list) == 0:
            return None
        else:
            for move_list in self.roxanne_table:
                random.shuffle(move_list)
                for move in move_list:
                    if move in action_list:
                        return move

    def get_move(self, board):
        """
        采用Roxanne 策略进行搜索
        :return: 落子
        """

        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        # print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        action = self.roxanne_select(board)
        return action


class AIPlayer(object):
    ''' 蒙特卡罗树搜索智能算法 '''

    def __init__(self, color, time_limit=2):
        """
        蒙特卡洛树搜索策略初始化
        :param color: 执棋方

        :param time_limit: 蒙特卡洛树搜索每步的搜索时间步长
        :param tick:记录开始搜索的时间
        :param sim_black, sim_white: 采用Roxanne策略代替随机策略搜索
        """

        self.time_limit = time_limit
        self.tick = 0
        self.sim_black = RoxannePlayer('X')
        self.sim_white = RoxannePlayer('O')
        self.color = color

    def mcts(self, board):
        """
        蒙特卡洛树搜索，在时间限制范围内，拓展节点搜索结果
        :return: 选择最佳拓展
        """

        root = TreeNode(None, self.color)

        # 设定一个时间停止计算，限定规模
        while time() - self.tick < self.time_limit - 1:
            sim_board = deepcopy(board)
            choice = self.select(root, sim_board)
            self.expand(choice, sim_board)
            winner, diff = self.simulate(choice, sim_board)
            back_score = [1, 0, 0.5][winner]
            if choice.color == 'X':
                back_score = 1 - back_score
            self.back_prop(choice, back_score)

        best_n = -1
        best_move = None
        for k in root.child.keys():
            if root.child[k].n > best_n:
                best_n = root.child[k].n
                best_move = k
        return best_move

    def select(self, node, board):
        """
        蒙特卡洛树搜索，节点选择
        :return: 搜索树向下递归选择子节点
        """

        if len(node.child) == 0:
            return node
        else:
            best_score = -1
            best_move = None
            for k in node.child.keys():
                if node.child[k].n == 0:
                    best_move = k
                    break
                else:
                    N = node.n
                    n = node.child[k].n
                    w = node.child[k].w
                    # 随着访问次数的增加，加号后面的值越来越小，因此我们的选择会更加倾向于选择那些还没怎么被统计过的节点
                    # 避免了蒙特卡洛树搜索会碰到的陷阱——一开始走了歪路。
                    score = w / n + sqrt(2 * log(N) / n)
                    if score > best_score:
                        best_score = score
                        best_move = k
            board._move(best_move, node.color)
            return self.select(node.child[best_move], board)

    def expand(self, node, board):
        """
        蒙特卡洛树搜索，节点扩展
        """

        for move in board.get_legal_actions(node.color):
            node.child[move] = TreeNode(node, oppo(node.color))

    def simulate(self, node, board):
        """
        蒙特卡洛树搜索，采用Roxanne策略代替随机策略搜索，模拟扩展搜索树
        """

        if node.color == 'O':
            current_player = self.sim_black
        else:
            current_player = self.sim_white
        sim_game = SilentGame(self.sim_black, self.sim_white, board, current_player)
        return sim_game.run()

    def back_prop(self, node, score):
        """
        蒙特卡洛树搜索，反向传播，回溯更新模拟路径中的节点奖励
        """

        node.n += 1
        node.w += score
        if node.parent is not None:
            self.back_prop(node.parent, 1 - score)

    def get_move(self, board):
        """
        蒙特卡洛树搜索
        :return: 采取最佳拓展落子策略
        """

        self.tick = time()
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        # print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        action = self.mcts(deepcopy(board))
        return action

## 测试AI玩家
# if __name__ == '__main__':
#     # 导入黑白棋文件
#     from game import Game
#
#     # 人类玩家黑棋初始化
#     black_player = HumanPlayer("X")
#
#     # AI 玩家 白棋初始化
#     white_player = AIPlayer("O")
#
#     # 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
#     game = Game(black_player, white_player)
#
#     # 开始下棋
#     game.run()
