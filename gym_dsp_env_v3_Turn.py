import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np
import networkx as nx


class dspEnv(gym.Env):
    """
    Graph-walk Gymnasium env.

    Node attributes used (conventions, not hard requirements):
      - 'label': any token (non-tour route grid: '.', route grid index: int, goal grid: letter, wall/unwalkable: '#')
      - 'is_goal': optional bool. If absent, falls back to "label is single letter".
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, G, start_node, goal_nodes, dis_nodes = [],
                 max_steps=200,
                 seed = 12345,
                 reward_step=-1.0, reward_invalid=-100.0, backtrack_Penalty = -50, quit_reward = -50,
                 render_mode="ansi",
                 control_mode = "allo",
                 is_touring = False, is_tourTesting = False, tourSeq = [], totalEpisode = 100
                 ):
        
        assert G is not None, "Provide a map graph G"

        self.iniG = G
        self.G = self.iniG.copy()
        self._build_dir_neighbor_map() ## Setup to enable turning
        self.control_mode = control_mode ## Setup to enable turning
        self.start_node = start_node
        self.dist2Goals = nx.multi_source_dijkstra_path_length(G, sources=goal_nodes)
        self.iniGoals = goal_nodes
        self.iniDisGoals = dis_nodes
        self.goal_nodes = self.iniGoals.copy() + self.iniDisGoals.copy()
        self.dis_nodes = self.iniDisGoals.copy()
        self.seed = seed
        self.render_mode = render_mode
        self.is_touring = is_touring
        self.is_tourTesting = is_tourTesting
        self.tourSeq = tourSeq
        self.tourInd = 0
        self.total_epi = totalEpisode
        self.curr_epi = -1

        self.termReward = quit_reward
        self.btPenalty = backtrack_Penalty
        self.backtrack = False
        self.btp = 0

        self.prev_a = None

        # Stable node index for Discrete observation
        self._node_list = sorted(self.G.nodes, key=lambda n: str(n))
        self._node_index = {n: i for i, n in enumerate(self._node_list)}
        self._goal_index = {g: i for i, g in enumerate(sorted(self.goal_nodes, key=lambda n: str(n)))}
        self._NGoal = len(self.goal_nodes)

        self.observation_space = spaces.MultiDiscrete([len(self._node_list), len(self._node_list) + 1, 4]) ## Setup to enable turning
                                                    # [current_node_idx, nearest_goal_idx_or_N, heading]
                                                    # current_node_idx: 0..N-1
                                                    # nearest_goal_idx: 0..N-1, or N meaning "no goal"
                                                    # heading: 0..3

        self.action_space = spaces.Discrete(5) 

        self.action_log = 0

        self.max_steps = int(max_steps)
        self.reward_step = float(reward_step)
        self.reward_invalid = float(reward_invalid)

        # self.np_random = 0  # set by reset
        self.curr = None
        self.prev = None
        self.curr_reward = 0
        self._step_count = 0

    # ---------- helpers ----------

    def _obs(self, node): ## Setup to enable turning
        N = len(self._node_list)

        curr_node_index = self._node_index[node]

        ng = self._get_nearest_goal()
        nearest_goal_index = self._node_index[ng] if ng is not None else N

        heading = getattr(self, "heading", 0)  # safe default if not set yet

        # return np.array([curr_node_index, nearest_goal_index, heading], dtype=np.int64)
        return np.array([curr_node_index, self.dist2Goals.get(node), heading], dtype=np.int64)

    
    def _label(self, node):
        return self.G.nodes[node].get("label")
    
    def _reward(self, node):
        return self.G.nodes[node].get("reward")

    def _recompute_nearest_goal_map(self, goalnodes):
        goalnodes = list(goalnodes) 
        if not goalnodes:
            self.dist_to_nearest_goal = {}
            self.paths_to_nearest_goal = {}
            self._has_goals = False
            return

        self.dist_to_nearest_goal, self.paths_to_nearest_goal = nx.multi_source_dijkstra(
            self.G, goalnodes, weight="weight"
        )

    def _is_goal(self, node, goalnodes):
        if node in self.dis_nodes:
            nx.set_node_attributes(self.G, {node: {"reward": 0}})
            self.dis_nodes.remove(node)
            self._recompute_nearest_goal_map(self.dis_nodes)

            
        elif node in goalnodes:
            nx.set_node_attributes(self.G, {node: {"reward": 0}})
            goalnodes.remove(node)
            self._recompute_nearest_goal_map(goalnodes)
            return True
    
    def _get_nearest_goal(self):
        path = self.paths_to_nearest_goal.get(self.curr)
        return path[0] if path else None
        
    def _stop_invalid(self, currentNode, currentStep):
        if (self.dist2Goals.get(currentNode) > (self.max_steps - currentStep)):
            return True

    # ---------- Turning mode ---------- ## Setup to enable turning

    def _build_dir_neighbor_map(self):
        # pos = [node for node in self.G.nodes]
        self.dir_nbr = {}
        for u in self.G.nodes:
            ux, uy = u
            m = {}
            for v in self.G.neighbors(u):
                vx, vy = v
                dx, dy = vx - ux, vy - uy
                if (dx, dy) == (0, -1): m[0] = v
                elif (dx, dy) == (1, 0): m[1] = v
                elif (dx, dy) == (0, 1): m[2] = v
                elif (dx, dy) == (-1, 0): m[3] = v
            self.dir_nbr[u] = m

    # egocentric meaning for 0..3
    

    def _propose_transition(self, a: int):
        next_node = self.curr
        moved = False
        invalid = False

        if self.control_mode == "allo":
            # 0..3 are N/E/S/W
            if a in (0, 1, 2, 3):
                nxt = self.dir_nbr[self.curr].get(a)
                if nxt is None:
                    invalid = True
                else:
                    next_node = nxt
                    moved = True
                    self.heading = a  # optional, but useful for logging/comparability
            else:
                invalid = True

        elif self.control_mode == "ego":
            TURN_L, TURN_R, FWD, BACK, STOP = 0, 1, 2, 3, 4
            # 0..3 are turn/turn/fwd/back
            if a == TURN_L:
                self.heading = (self.heading - 1) % 4
            elif a == TURN_R:
                self.heading = (self.heading + 1) % 4
            elif a in (FWD, BACK):
                move_h = self.heading if a == FWD else (self.heading + 2) % 4
                nxt = self.dir_nbr[self.curr].get(move_h)
                if nxt is None:
                    invalid = True
                else:
                    next_node = nxt
                    moved = True
            else:
                invalid = True

        else:
            raise ValueError(f"Unknown control_mode: {self.control_mode}")

        return next_node, moved, invalid


    # ---------- gym api ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_epi += 1
        self.G = self.iniG.copy()
        self.curr = self.start_node
        self.prev = self.start_node
        self._step_count = 0 
        self.goal_nodes = self.iniGoals.copy() + self.iniDisGoals.copy()
        self.dis_nodes = self.iniDisGoals.copy()
        self.tourInd = 0
        self._recompute_nearest_goal_map(self.goal_nodes+self.dis_nodes)
        self.heading = 0 ## Setup to enable turning

        nbrs = list(self.G.neighbors(self.curr))
        # # Basic sanity: start must be part of the graph and have an index
        if self.curr not in self._node_index:
            raise ValueError("start_node not present in graph nodes.")

        obs = self._obs(self.curr)
        info = {#"G": list(self.G),
            "node": self.curr,
            "label": self._label(self.curr),
            "neighbors": nbrs,
            "valid_actions": list(range(len(nbrs))),
        }
        return obs, info

    def step(self, action):
        self.backtrack = False
        self._step_count += 1

        terminated = False
        truncated = False
        reward = 0.0

        a = int(action)
        self.action_log = a

        invalid = False
        moved = False
        self.endReward = 0

        next_node = self.curr

        # ---------- STOP action (a==4) ----------
        if a == 4:
            if self.is_touring:
                # match your touring STOP behavior
                if (str(self._label(self.curr)) == str(0)) and (self.tourInd == len(self.tourSeq) - 1):
                    terminated = True
                else:
                    reward = -100
                next_node = self.curr
                moved = False

            else:
                # match your "half invalid" penalty unless done
                if not (self.goal_nodes + self.dis_nodes):
                    terminated = True
                else:
                    reward += 0.5 * self.reward_invalid
                next_node = self.curr
                moved = False

        # ---------- Non-STOP actions ----------
        else:
            next_node, moved, invalid = self._propose_transition(a)

            if self.is_touring:
                # enforce tour sequence on movement
                if not moved:
                    # strict mimic of original touring: anything not advancing is invalid
                    invalid = True
                else:
                    if self.tourInd < len(self.tourSeq) - 1:
                        if str(self._label(next_node)) != str(self.tourSeq[self.tourInd + 1]):
                            invalid = True
                    else:
                        invalid = True

            if invalid:
                # punish and stay stuck
                reward = self.reward_invalid if self.is_touring else (reward + self.reward_invalid)
                next_node = self.curr
                moved = False
            else:
                if self.is_touring and moved:
                    reward = 10
                    self.tourInd = min(self.tourInd + 1, len(self.tourSeq) - 1)

        # ---------- backtrack logic (keep your style) ----------
        self.backtrack = (self.prev == next_node)
        self.btp = self.btPenalty if self.backtrack else 0
        reward += self.btp

        self.prev = self.curr
        self.curr = next_node
        self.prev_a = a

        reward += self._reward(self.curr) + self.reward_step

        if terminated:
            reward = self.termReward

        self.curr_reward = reward

        if moved:
            self._is_goal(self.curr, self.goal_nodes)

            if self.tourInd == len(self.tourSeq) - 1:
                terminated = True

            if not (self.is_touring | self.is_tourTesting):
                if not (self.goal_nodes + self.dis_nodes):
                    terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._obs(self.curr)

        # optional: valid actions depends on control mode
        if self.control_mode == "allo":
            valid_actions = [d for d in (0, 1, 2, 3) if self.dir_nbr[self.curr].get(d) is not None] + [4]
        else:
            valid_actions = [0, 1, 4]  # turns + stop always valid
            if self.dir_nbr[self.curr].get(self.heading) is not None:
                valid_actions.append(2)
            if self.dir_nbr[self.curr].get((self.heading + 2) % 4) is not None:
                valid_actions.append(3)

        info = {
            "node": self.curr,
            "label": self._label(self.curr),
            "heading": getattr(self, "heading", None),
            "remainingGoal": self.goal_nodes,
            "reward": reward,
            "neighbors": list(self.G.neighbors(self.curr)),
            "valid_actions": valid_actions,
            "moved": moved,
            "prev_node": self.prev,
            "control_mode": self.control_mode,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return
        nbrs = list(self.G.neighbors(self.curr))
        nbr_labs = [self._label(n) for n in nbrs]


        goal_labels = {n: self._label(n) for n in self.goal_nodes+self.dis_nodes}
        inigoal_labels = {n: self._label(n) for n in self.iniGoals}
        s = f"[Episode {self.curr_epi}/{self.total_epi} step {self._step_count}] at {self._label(self.curr)} reward={self.curr_reward}, action {self.action_log} -> {list(zip(nbrs, nbr_labs))}"
        print(s)
        return s
    

if __name__ == "__main__":    
    import dspMapBuilder
    G, pos, goalnodes, startnode, _, routenodes = dspMapBuilder.loadMap()
    env = dspEnv(G, startnode, goalnodes)

    obs, info = env.reset(seed=0)
    done = False
    while True:
        # mask invalid actions if your algo supports it
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        
        print(env.render())
        if term or trunc:
            break
    