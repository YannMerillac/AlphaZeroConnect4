import connect4
import mcts
import mcts_base
import model
import torch

# create a connect4 game
game = connect4.Game()

model = model.Connect4Model(board_size=42, action_size=7, hidden_size=128, device="cpu")
#model.load_state_dict(torch.load('latest.pth')["state_dict"])

# create players
player1 = mcts_base.MCTS(iterations=100)
#player1 = mcts.MCTS(model, iterations=100)
player2 = mcts.MCTS(model, iterations=100)


# loop until the game is drawn
turn = 1
while len(game.moves()) > 0:
    game.show()

    # reverse these if you want MCTS to go first
    if game.turn == 1:
        print("-- Player 1 --")
        move, policy = player1.act()
    else:
        print("-- Player 2 --")
        move, policy = player2.act()
    #print(game.board, policy)
    # update the internal state of both players
    player1.feed(move)
    player2.feed(move)

    # if the move wins the game, then break
    if game.make_move(move):
        if turn == 1:
            print("Player 1 won")
        else:
            print("Player 2 won")
        break
    turn *= -1
game.show()
