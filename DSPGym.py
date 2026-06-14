import networkx as nx
import pandas as pd
import dspMapBuilder
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import os
from functools import partial
from pathlib import Path

from tqdm.notebook import trange, tqdm


from gymnasium.envs.registration import register

# Resolved relative to this file so the module works from any CWD.
_HERE    = Path(__file__).resolve().parent
MAPS_DIR = _HERE / "maps_and_settings"

# gymPy = "gym_dsp_env_v2_multiGoal"
gymPy = "gym_dsp_env_v3_Turn"

register(
    id=gymPy,
    entry_point=f"{gymPy}:dspEnv"
)

# -----------------------------
# Small utilities
# -----------------------------

def get_tour_pair(map_type: str, pair: str):
    if "dsp1" in pair:
        return "dsp1_tour"
    if "dsp2" in pair:
        return "dsp2_tour"
    if "Gal" in map_type:
        return "gal_tour"
    return None


def load_tour_seq(map_type: str, pair: str, csv_path=None):
    tourPair = get_tour_pair(map_type, pair)
    if tourPair is None:
        return None
    if csv_path is None:
        csv_path = MAPS_DIR / "DSPTourSeq.csv"
    return pd.read_csv(csv_path)[tourPair].dropna().values


def epsilon_schedule(e0, e1, ep, total_eps, curve=2):
    x = (ep / total_eps) ** curve
    return e0 * (e1 / e0) ** x


def softmax_sample(q_vals, temperature):
    # numerical stability
    z = q_vals - np.max(q_vals)
    probs = np.exp(z / max(temperature, 1e-8))
    probs /= probs.sum()
    return int(np.random.choice(len(q_vals), p=probs))

def show_plot(block=False):
    """Non-blocking plt.show for scripts; still works in notebooks."""
    try:
        plt.show(block=block)
        plt.pause(0.001)
    except TypeError:
        # some environments don't support show(block=...)
        plt.show()


def plot_training_curves(steps_per_episode, reward_per_episode, epsilon_log, max_steps, goal_found=None):
    rmin, rmax = float(np.min(reward_per_episode)), float(np.max(reward_per_episode))
    emin, emax = float(np.min(epsilon_log)), float(np.max(epsilon_log))
    denom = (rmax - rmin) if (rmax - rmin) != 0 else 1.0
    edenom = (emax - emin) if (emax - emin) != 0 else 1.0
    reward_scaled = [(x - rmin) / denom * float(np.max(steps_per_episode)) for x in reward_per_episode]
    epsilon_scaled = [(x - emin) / edenom * float(np.max(steps_per_episode)) for x in epsilon_log]

    plt.plot(steps_per_episode, label="Steps")
    plt.plot(np.array(epsilon_scaled), label="Epsilon/Temp (scaled)")
    plt.plot(np.array(reward_scaled), label="Reward (scaled)")
    if goal_found is not None:
        plt.plot(np.array(goal_found) * 10, label="Goal found (scaled)")
    plt.legend()
    show_plot(block=False)


def plot_eval_route(G, goalnodes, disNodes, routenodes, startnode, route_idx):
    node_index = {i: n for i, n in enumerate(sorted(G.nodes, key=lambda n: str(n)))}

    plt.figure(figsize=(5, 5))
    pos_plot = {node: (node[1], -node[0]) for node in G.nodes()}

    # nodes
    nx.draw_networkx_nodes(G, pos_plot, nodelist=goalnodes, node_color="lightgreen", node_size=100)
    nx.draw_networkx_nodes(G, pos_plot, nodelist=disNodes, node_color="cyan", node_size=100)
    nx.draw_networkx_nodes(G, pos_plot, nodelist=routenodes, node_color="yellow", node_size=20)
    nx.draw_networkx_nodes(G, pos_plot, nodelist=[startnode], node_color="red", node_size=100)

    # edges + labels
    nx.draw_networkx_edges(G, pos_plot)
    nx.draw_networkx_labels(G, pos_plot, nx.get_node_attributes(G, "label"), font_size=10)

    # arrows
    for i in range(len(route_idx) - 1):
        start = pos_plot[node_index[route_idx[i]]]
        end = pos_plot[node_index[route_idx[i + 1]]]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(
            start[0], start[1],
            dx * 0.85, dy * 0.85,
            head_width=0.5, head_length=0.51,
            fc="green", ec="green",
            length_includes_head=True
        )

    plt.title(f"Maze Graph with Learned Policy Path, Steps:{len(route_idx)}")
    plt.axis("off")
    show_plot(block=False)


# -----------------------------
# Environment builder
# -----------------------------
def build_env(
    phase: str,
    envScript,
    G,
    startnode,
    goalnodes,
    disNodes,
    max_steps,
    stepPenalty,
    episodes,
    backtrack_Penalty=None,
    quit_reward=None,
    tourSeq=None,
    testTourRoute=False,
    control_mode = 'allo'
):
    """
    phase: "tour" | "train" | "eval"
    """
    if gymPy == "gym_dsp_env_v3_Turn":
        if phase == "tour":
            return gym.make(
                envScript,
                G=G, start_node=startnode, goal_nodes=goalnodes, dis_nodes=disNodes,
                max_steps=max_steps, reward_step=stepPenalty,
                is_touring=True, tourSeq=tourSeq, control_mode = control_mode,
                totalEpisode=episodes
            )

        if phase == "train":
            return gym.make(
                envScript,
                G=G, start_node=startnode, goal_nodes=goalnodes, dis_nodes=disNodes,
                max_steps=max_steps, reward_step=stepPenalty,
                backtrack_Penalty=backtrack_Penalty,
                quit_reward=quit_reward, control_mode = control_mode,
                totalEpisode=episodes
            )

        if phase == "eval":
            return gym.make(
                envScript,
                G=G, start_node=startnode, goal_nodes=goalnodes, dis_nodes=disNodes,
                max_steps=max_steps, reward_step=stepPenalty,
                is_tourTesting=testTourRoute, tourSeq=tourSeq, control_mode = control_mode,
                totalEpisode=episodes
            )

        raise ValueError(f"Unknown phase='{phase}'")
    else:
        if phase == "tour":
            return gym.make(
                envScript,
                G=G, start_node=startnode, goal_nodes=goalnodes, dis_nodes=disNodes,
                max_steps=max_steps, reward_step=stepPenalty,
                is_touring=True, tourSeq=tourSeq, #control_mode = control_mode,
                totalEpisode=episodes
            )

        if phase == "train":
            return gym.make(
                envScript,
                G=G, start_node=startnode, goal_nodes=goalnodes, dis_nodes=disNodes,
                max_steps=max_steps, reward_step=stepPenalty,
                backtrack_Penalty=backtrack_Penalty,
                quit_reward=quit_reward, #control_mode = control_mode,
                totalEpisode=episodes
            )

        if phase == "eval":
            return gym.make(
                envScript,
                G=G, start_node=startnode, goal_nodes=goalnodes, dis_nodes=disNodes,
                max_steps=max_steps, reward_step=stepPenalty,
                is_tourTesting=testTourRoute, tourSeq=tourSeq, #control_mode = control_mode,
                totalEpisode=episodes
            )

        raise ValueError(f"Unknown phase='{phase}'")


# -----------------------------
# Q init / load logic
# -----------------------------
def init_Q(
    phase: str,
    G,
    n_actions: int,
    useTour=True,
    default_Q_Strength=1.0,
    Qtour_in_memory=None,
    Qtable_in_memory=None,
    testTourRoute=False,
    TestQtour=False,
):
    """
    No disk IO version.
    """
    if gymPy == "gym_dsp_env_v3_Turn":

        N = len(G)
        H = 4                 # headings
        NG = N + 1            # nearest-goal index includes sentinel N = "no goal"

        if phase == "tour":
            # state = (node_idx, tourInd, heading)
            return np.zeros((N, 256, H, n_actions), dtype=np.float32)

        if phase == "train":
            # state = (node_idx, nearest_goal_idx_or_N, heading)
            if useTour and Qtour_in_memory is not None:
                Q0 = Qtour_in_memory

                # If old table had no heading dim, expand by copying across headings
                # expected old shapes: (N, 256, n_actions) or (N, NG, n_actions)
                if Q0.ndim == 3:
                    Q0 = Q0[:, :, None, :]          # add heading axis
                    Q0 = np.repeat(Q0, H, axis=2)   # tile across headings

                return Q0 * default_Q_Strength

            Q = np.zeros((N, NG, H, n_actions), dtype=np.float32)
            return Q

    else:
        
        if phase == "tour":
            return np.zeros((len(G), 256, n_actions))

        if phase == "train":
            if useTour and Qtour_in_memory is not None:
                return Qtour_in_memory * default_Q_Strength

            Q = np.zeros((len(G), len(G), n_actions))
            Q[:, :, 0 : n_actions - 1] += 10  # encourage exploration except QUIT
            return Q

    if phase == "eval":
        # Decide what Q to use
        if TestQtour:
            if Qtour_in_memory is None:
                raise ValueError("Eval requested Qtour, but Qtour_in_memory=None")
            return Qtour_in_memory

        if testTourRoute:
            if Qtour_in_memory is None:
                raise ValueError("Eval requested tour-route Q, but Qtour_in_memory=None")
            return Qtour_in_memory

        # otherwise normal eval uses trained Q
        if Qtable_in_memory is None:
            raise ValueError("Eval requested Qtable, but Qtable_in_memory=None")
        return Qtable_in_memory

    raise ValueError(f"Unknown phase='{phase}'")


# -----------------------------
# Action selection
# -----------------------------
def select_action(
    phase: str,
    env,
    Q,
    state,
    epsilon,
    temp_cur,
    p_momentum,
    prev_action,
    prev_action_invalid,
):
    """
    Uses your original rules:
      - tour: epsilon random else argmax
      - train: softmax with temp=temp_cur (caller owns the schedule)
      - eval: argmax
    """

    # momentum first
    if (
        prev_action is not None
        and prev_action != 4
        and not prev_action_invalid
        and random.random() < p_momentum
    ):
        return prev_action

    q_state_idx = tuple(state)

    if phase == "tour":
        if random.random() < epsilon:
            return env.action_space.sample()
        return int(np.argmax(Q[q_state_idx]))

    if phase == "train":
        q_vals = Q[q_state_idx]
        return softmax_sample(q_vals, temp_cur)

    if phase == "eval":
        return int(np.argmax(Q[q_state_idx]))

    raise ValueError(f"Unknown phase='{phase}'")


# -----------------------------
# Q update
# -----------------------------
def update_Q(
    phase: str,
    Q,
    state,
    action,
    new_state,
    reward,
    done,
    alpha,
    gamma,
):
    """
    tour update (your exact math):
      Q += alpha*(gamma*max(Q[s']) - Q[s,a] + r)

    train update (your exact math):
      Q_tar = r if done else gamma*max(Q[s']) - Q[s,a] + r
      Q += alpha*Q_tar
    """
    if phase not in ("tour", "train"):
        return  # eval: no learning

    q_state_idx = tuple(state)
    q_new_state_idx = tuple(new_state)
    q_state_action_idx = q_state_idx + (action,)

    if phase == "tour":
        Q[q_state_action_idx] = Q[q_state_action_idx] + alpha * (
            gamma * np.max(Q[q_new_state_idx]) - Q[q_state_action_idx] + reward
        )
        return

    # train
    Q_tar = reward if done else gamma * np.max(Q[q_new_state_idx]) - Q[q_state_action_idx] + reward
    Q[q_state_action_idx] = Q[q_state_action_idx] + alpha * (Q_tar)


# -----------------------------
# Main loop runner for any phase
# -----------------------------
def run_phase_loop(
    phase: str,
    env,
    Q,
    episodes,
    max_steps,
    alpha,
    gamma,
    epsilon,
    temperature,
    p_momentum,
    epsilon_end,
    epsilon_curve,
    temp_end=None,
    temp_curve=2,
    rev_strength = 1,
    render=False,
    print_progress=False,
):
    steps_per_episode = np.zeros(episodes)
    reward_per_episode = np.zeros(episodes)
    goal_found_per_episode = np.zeros(episodes)# if phase == "train" else None
    route_per_episode = defaultdict(list)
    epsilon_log = [epsilon]
    eps_cur = epsilon
    temp_cur = temperature  # tracks current effective softmax temperature

    total_steps = 0
    total_reward = 0.0
    lastRewards = np.zeros(10)
    rm = 0

    for ep in tqdm(range(episodes), disable = not all([(phase == "train"), print_progress])):
        state, _ = env.reset()
        done = False

        prev_action = None
        prev_action_invalid = False

        route_idx_list = [state[0]]
        goal_found = 0
        ep_trajectory = []

        while not done:
            action = select_action(
                phase=phase,
                env=env,
                Q=Q,
                state=state,
                epsilon=eps_cur,
                temp_cur=temp_cur,
                p_momentum=p_momentum,
                prev_action=prev_action,
                prev_action_invalid=prev_action_invalid,
            )

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            prev_action_invalid = (set(state) == set(new_state))
            prev_action = action

            route_idx_list.append(new_state[0])
            ep_trajectory.append((tuple(state), action, tuple(new_state)))

            # learning update
            update_Q(
                phase=phase,
                Q=Q,
                state=state,
                action=action,
                new_state=new_state,
                reward=reward,
                done=done,
                alpha=alpha,
                gamma=gamma,
            )

            state = new_state
            total_steps += 1
            total_reward += float(reward)

            if phase == "train":
                goal_found += 1 if float(reward) > 0 else 0

            if render:
                try:
                    env.render()
                except Exception:
                    pass

            if done:
                steps_per_episode[ep] = total_steps
                reward_per_episode[ep] = total_reward
                route_per_episode[ep] = route_idx_list

                if phase == "train":
                    goal_found_per_episode[ep] = goal_found

                reverse_action_map = {
                    0: 1,   # clockwise  -> counter-clockwise
                    1: 0,   # counter-clockwise -> clockwise
                    # leave other actions (QUIT etc.) unchanged
                }

                # ---- backward sweep (tour only) ----
                if phase == "tour" and reverse_action_map is not None:
                    reverse_goal_node = ep_trajectory[0][0][0]  # original start
                    for t in reversed(range(len(ep_trajectory) - 1)):
                        s, a, s_next = ep_trajectory[t]
                        a_rev = reverse_action_map.get(a)
                        if a_rev is None:
                            continue
                        r_rev = reward if s[0] != reverse_goal_node else 0.0
                        # Clamp effective reversal LR < 1.0: rev_strength * alpha >= 1
                        # causes the TD update to overshoot, leading to Q divergence → NaN.
                        rev_lr = min(rev_strength * alpha, 0.99)
                        Q[s_next + (a_rev,)] += rev_lr * (
                            gamma * np.max(Q[s]) - Q[s_next + (a_rev,)] + r_rev
                        )
                    ep_trajectory = []  # clear for next episode
                # ------------------------------------
                
                rm = rm + 1 if rm < len(lastRewards)-1 else 0
                lastRewards[rm] = total_reward

                total_steps = 0
                total_reward = 0.0

        # epsilon schedule (tour/train only)
        if phase in ("tour", "train"):
            eps_cur = epsilon_schedule(epsilon, epsilon_end, ep, episodes, curve=epsilon_curve)
            epsilon_log.append(eps_cur)
            # temperature schedule: independent of epsilon when temp_end is given;
            # falls back to eps_cur * temperature for backward compatibility
            if temp_end is not None:
                temp_cur = epsilon_schedule(temperature, temp_end, ep, episodes, curve=temp_curve)
            else:
                temp_cur = eps_cur * temperature

        if len(set(lastRewards)) == 1:
            break

    return {
        "Q": Q,
        "steps_per_episode": np.trim_zeros(steps_per_episode),
        "reward_per_episode": np.trim_zeros(reward_per_episode),
        "goal_found_per_episode": np.trim_zeros(goal_found_per_episode),
        "route_per_episode": route_per_episode,
        "epsilon_log": epsilon_log,
    }

    # -----------------------------
# Full pipeline entry point
# -----------------------------
def run_q(
    Tour=False,
    Eval=True,
    Train=True,

    tourEpisodes=10,
    episodes=500,
    evalEpisodes=1,

    max_steps=200,

    map_type="OG",
    pair="dsp2_07",

    derailPenalty=-1,
    goalReward=100,
    disReward=50,

    stepPenalty=-1,
    backtrack_Penalty=-5,
    quit_reward=-50,

    envScript=None,

    # train hyperparams
    temperature=5,
    temp_end=None,
    temp_curve=2,
    alpha=0.05,
    gamma=0.95,
    epsilon=1.0,
    epsilon_end=0.001,


    # tour hyperparams
    tour_alpha=0.7,
    tour_gamma=0.1,
    tour_epsilon=1.0,
    rev_strength=1, # The strength of reversal HER after a tour, basically force agent to reverse the tour.

    p_momentum=0.0,
    
    control_mode = "allo",

    useTour=False,
    default_Q_Strength=1.0,

    testTourRoute=False,
    TestQtour=False,

    show_train_progress=True,

    plotting=False,
    render=False,
    save=True,
):
    if envScript is None:
        envScript = gymPy

    results = {}

    Qtour_mem = None
    Qtrain_mem = None

    # -------------------------
    # TOUR PHASE
    # -------------------------
    if useTour:
        tourSeq = load_tour_seq(map_type, pair)

        G, pos, goalnodes, startnode, disNodes, routenodes = dspMapBuilder.loadMap(
            map_type, pair, derailPenalty, goalReward, disReward, is_tour=True
        )

        env = build_env(
            phase="tour",
            envScript=envScript,
            G=G,
            startnode=startnode,
            goalnodes=goalnodes,
            disNodes=disNodes,
            max_steps=max_steps,
            stepPenalty=stepPenalty,
            episodes=tourEpisodes,
            tourSeq=tourSeq,
            control_mode = control_mode
        )

        n_actions = env.action_space.n  # type: ignore
        Q = init_Q("tour", G, n_actions)

        tour_out = run_phase_loop(
            phase="tour",
            env=env,
            Q=Q,
            episodes=tourEpisodes,
            max_steps=max_steps,
            alpha=tour_alpha,
            gamma=tour_gamma,
            rev_strength=rev_strength,
            epsilon=tour_epsilon,
            temperature=temperature,
            p_momentum=p_momentum,
            epsilon_end=epsilon_end,
            epsilon_curve=1,
            render=render,
            print_progress= False,
        )

        env.close()
        Qtour_mem = tour_out["Q"]

    # -------------------------
    # TRAIN PHASE
    # -------------------------
    if Train:
        G, pos, goalnodes, startnode, disNodes, routenodes = dspMapBuilder.loadMap(
            map_type, pair, derailPenalty, goalReward, disReward
        )

        env = build_env(
            phase="train",
            envScript=envScript,
            G=G,
            startnode=startnode,
            goalnodes=goalnodes,
            disNodes=disNodes,
            max_steps=max_steps,
            stepPenalty=stepPenalty,
            episodes=episodes,
            backtrack_Penalty=backtrack_Penalty,
            quit_reward=quit_reward,
            control_mode = control_mode
        )

        n_actions = env.action_space.n  # type: ignore

        Q = init_Q(
            phase="train",
            G=G,
            n_actions=n_actions,
            useTour=useTour,
            default_Q_Strength=default_Q_Strength,
            Qtour_in_memory=Qtour_mem
        )

        # NOTE: this keeps your original training "epsilon -> 1" schedule
        train_out = run_phase_loop(
            phase="train",
            env=env,
            Q=Q,
            episodes=episodes,
            max_steps=max_steps,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            temperature=temperature,
            temp_end=temp_end,
            temp_curve=temp_curve,
            p_momentum=p_momentum,
            epsilon_end=epsilon_end,
            epsilon_curve=2,
            render=render,
            print_progress=show_train_progress,
        )

        env.close()
        results["train"] = train_out
        Qtrain_mem = train_out["Q"]

        if plotting:
            plot_training_curves(
                train_out["steps_per_episode"],
                train_out["reward_per_episode"],
                train_out["epsilon_log"],
                max_steps,
                goal_found=train_out["goal_found_per_episode"],
            )

    # -------------------------
    # EVAL PHASE
    # -------------------------
    route_seq = []
    if Eval:
        tourSeq = load_tour_seq(map_type, pair)

        G, pos, goalnodes, startnode, disNodes, routenodes = dspMapBuilder.loadMap(
            map_type, pair, derailPenalty, goalReward, disReward, is_tour=testTourRoute
        )

        env = build_env(
            phase="eval",
            envScript=envScript,
            G=G,
            startnode=startnode,
            goalnodes=goalnodes,
            disNodes=disNodes,
            max_steps=max_steps,
            stepPenalty=stepPenalty,
            episodes=evalEpisodes,
            tourSeq=tourSeq,
            testTourRoute=testTourRoute,
            control_mode = control_mode
        )

        n_actions = env.action_space.n  # type: ignore

        Qeval = init_Q(
            phase="eval",
            G=G,
            n_actions=n_actions,
            Qtour_in_memory=Qtour_mem,
            Qtable_in_memory=Qtrain_mem,
            testTourRoute=testTourRoute,
            TestQtour=TestQtour,
        )

        eval_out = run_phase_loop(
            phase="eval",
            env=env,
            Q=Qeval,
            episodes=evalEpisodes,
            max_steps=max_steps,
            alpha=0.0,              # not used
            gamma=0.0,              # not used
            epsilon=0.0,            # not used
            temperature=temperature,
            p_momentum=0.0,         # momentum off in eval
            epsilon_end=0.0,
            epsilon_curve=1,
            render=render,
            print_progress=False,
        )

        env.close()

        # plot route for episode 0 (same as your eval_q)
        route_idx = eval_out["route_per_episode"][0]
        nodes = sorted(G.nodes, key=str)
        route_seq = {"RouteNodeIdx": np.array(eval_out["route_per_episode"][0]).tolist(),
                     "RouteNodeLabel": [G.nodes[nodes[i]].get("label") for i in route_idx]}
        if plotting:
            plot_eval_route(G, goalnodes, disNodes, routenodes, startnode, route_idx)

    # return results
    return route_seq
