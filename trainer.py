import os
import numpy as np
from random import shuffle
import connect4
import mcts
import torch
import torch.optim as optim


class Trainer:

    def __init__(self, model, args):
        self.model = model
        self.args = args

    def execute_episode(self):
        game_states = []
        game_policies = []
        player_turns = []
        game_rewards = None
        # create a connect4 game
        game = connect4.Game()
        # create players
        player1 = mcts.MCTS(self.model, iterations=self.args["num_simulations"])
        player2 = mcts.MCTS(self.model, iterations=self.args["num_simulations"], id=-1)

        # loop until the game is drawn
        turn = 1
        while len(game.moves()) > 0:
            if game.turn == 1:
                current_player = player1
            else:
                current_player = player2
            move, policy = current_player.act()

            game_states.append(current_player.id * game.board.ravel())
            game_policies.append(policy)
            # update the internal state of both players
            player1.feed(move)
            player2.feed(move)
            player_turns.append(turn)
            # if the move wins the game, then break
            if game.make_move(move):
                if turn == 1:
                    game_rewards = np.array(player_turns)
                else:
                    game_rewards = -np.array(player_turns)
                break

            turn *= -1
        game.show()
        if game_rewards is None:
            game_rewards = [0] * len(game_policies)
        return [ep for ep in zip(game_states, game_policies, game_rewards)]

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):

            print("{}/{}".format(i, self.args['numIters']))

            train_examples = []

            for eps in range(self.args['numEps']):
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                #boards = boards.contiguous().cuda()
                #target_pis = target_pis.contiguous().cuda()
                #target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)
