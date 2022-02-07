import connect4
import mcts

# create a connect4 game
game = connect4.Game()

# create players
player1 = mcts.MCTS(iterations=100)
player2 = mcts.MCTS(iterations=100)


# loop until the game is drawn
while len(game.moves()) > 0:
    game.show()

    # reverse these if you want MCTS to go first
    if game.turn == 1:
        move, policy = player1.act()
    else:
        move, policy = player2.act()
    print(game.board, policy)
    # update the internal state of both players
    player1.feed(move)
    player2.feed(move)

    # if the move wins the game, then break
    if game.make_move(move):
        break
game.show()
