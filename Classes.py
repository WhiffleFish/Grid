import numpy as np
import matplotlib.pyplot as plt

class gridAgent(object):
    
    def __init__(self,neg,random_state=None):
        self.num_neg = neg  # Number of "punishing" cells in grid
        self.random_state = random_state # For reproducibility
        self.initialize_grid(self.num_neg, random_state)
        self.reset()

        # Used to calculate value of cells
        self.raw_exp_grid = np.zeros((10,10)) # cell wins - cell losses
        self.land_count_grid = np.zeros((10,10)) # How many times has agent landed on cell
        self.exp_grid = np.zeros((10,10)) # (cell wins - cell losses)/(times landed on cell)


    def reset(self):
        # Revert to beginning of game
        self.path = [self.path[0][:]]
        self.pos = self.path[0][:]
        self.isEnd = False
        self.score = 0
        
        
    def initialize_grid(self,neg,random_state):
        '''
        Set:
            - Initial agent position
            - Reward cell position
            - "Punishment" cell position
        '''
        self.path = []
        
        if random_state:
            np.random.seed(random_state)
        
        # Select Positions
        coords = []
        n = 0
        while n < neg+2:
            [x,y] = np.random.uniform(0,10,2).astype(int)
            if not [x,y] in coords:
                coords.append([x,y])
                n += 1

        self.pos = coords.pop()
        self.path.append(self.pos[:])
        self.reward_pos = coords.pop()
        self.negs = coords


    def draw_grid(self, figsize=(6,5)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        x = np.zeros((10,10,3))
        x[tuple([k[0] for k in self.negs]),tuple([k[1] for k in self.negs]),tuple([0]*len(self.negs))] = 1
        x[(*self.reward_pos,1)] = 1
        x[tuple([k[0] for k in self.path]),tuple([k[1] for k in self.path]),tuple([2]*len(self.path))] = 1
        ax.imshow(x)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(*self.pos[::-1], s=50, marker='+',c='red')
        
        
    def move(self,direction):
        assert not self.isEnd  # Can't move if the game has ended

        # Deterministic Input
        if direction == 'left' and self.pos[1] != 0:
            self.pos[1] -= 1
        elif direction == 'right' and self.pos[1] != 9:
            self.pos[1] += 1
        elif direction == 'up' and self.pos[0] != 0:
            self.pos[0] -= 1
        elif direction == 'down' and self.pos[0] != 9:
            self.pos[0] += 1
        else:
            self.score -= 1
            self.end() # Ran off board or gave invalid input
    
        self.path.append(self.pos[:])
        
        if self.pos in self.negs:
            self.score -= 1
            self.end()
        elif self.pos == self.reward_pos:
            self.score += 1
            self.end()
    
    
    def action(self, direction):
        # Introduces Stochastic element
        move_list = ['left','right','up','down']
        
        # Process input
        if np.random.uniform(0,1) <= 0.7:
            self.move(direction)
        else:
            move_list.remove(direction)
            self.move(np.random.choice(move_list))
    
    
    def policy(self):
        moves = ['up', 'down', 'left', 'right']

        # Pad with -2: disincentivize hopping off grid
        pad_grid = np.pad(self.exp_grid, pad_width=1, mode='constant', constant_values=-2)
        
        pad_pos = (self.pos[0]+1, self.pos[1]+1)

        # Determine scores of neighboring cells
        scores = np.array([pad_grid[pad_pos[0]-1,pad_pos[1]],
                           pad_grid[pad_pos[0]+1,pad_pos[1]],
                           pad_grid[pad_pos[0],pad_pos[1]-1],
                           pad_grid[pad_pos[0],pad_pos[1]+1]])
        
        # Where is the score maximized
        max_indices = np.argwhere(scores==scores.max())
        
        # Return random choice between blocks of equivalent max score
        return moves[np.random.choice(max_indices.flatten())]


    def end(self):
        # End current game cycle
        self.isEnd = True

        # Update cell values based on success or failure
        unique_moves = np.unique(self.path[:], axis=0)
        self.raw_exp_grid[tuple(unique_moves[:,0]),tuple(unique_moves[:,1])] += self.score
        self.land_count_grid[tuple(unique_moves[:,0]),tuple(unique_moves[:,1])] += 1

        self.exp_grid = np.divide(self.raw_exp_grid,self.land_count_grid, where=self.land_count_grid!=0)


    def play(self,n):

        for _ in range(n):
            self.reset()

            while not self.isEnd:
                move = self.policy()  # Plan Move
                self.action(move)  # Make Move