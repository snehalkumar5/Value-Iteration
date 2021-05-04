import numpy as np
import json
import copy
import os

MDL_TEAM = 79
ARR = [0.5, 1, 2]
Y = ARR[MDL_TEAM % 3]
STEP_COST = -10 / Y
DISCOUNT = 0.999
# DISCOUNT = 0.25
DELTA = 0.001

POSITIONS = {
    "WEST" : 0,
    "NORTH" : 1,
    "EAST" : 2,
    "SOUTH" : 3,
    "CENTER" : 4
}
IJ_POS = ['W', 'N', 'E', 'S', 'C']
HEALTH_VVALS = [0, 25, 50, 75, 100]
max_hp = 100
max_num_arr = 3
max_num_mats = 2
MM_STR = ['D', 'R']
STATES_MM = {
    'DORMANT':0,
    'READY':1
}
# REWARDS
STATE_REWARD = np.zeros((5, 3, 4, 2, max_hp + 1))
STATE_REWARD[:, :, :, :, 0] = 50
MM_HIT_R = -40

# ACTIONS
MOVE_CHOOSE = np.array(["UP", "DOWN", "LEFT", "RIGHT", "STAY"])
dormant={
    "STAY": 0.8,
    "GET_READY" : 0.2
}
ready={
    "ATTACK": 0.5,
    "STAY": 0.5
}

ACTIONS = {
    POSITIONS['CENTER']: {
        "UP": {
            "S": (0.85, POSITIONS['NORTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "DOWN": {
            "S": (0.85, POSITIONS['SOUTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "LEFT": {
            "S": (0.85, POSITIONS['WEST']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "RIGHT": {
            "S": (0.85, POSITIONS['EAST']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "HIT": {
            "S": 0.1,
            "F": 0.9
        },
        "STAY": {
            "S": (0.85, POSITIONS['CENTER']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "SHOOT": {"S": 0.5,"F": 0.5},
    },
    POSITIONS['NORTH']: {
        "DOWN": {
            "S": (0.85, POSITIONS['CENTER']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "STAY": {
            "S": (0.85, POSITIONS['NORTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "CRAFT": [0.5,0.35,0.15]
    },
    POSITIONS['SOUTH']: {
        "UP": {
            "S": (0.85, POSITIONS['CENTER']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "STAY": {
            "S": (0.85, POSITIONS['SOUTH']),
            "F": (0.15, POSITIONS['EAST'])
        },
        "GATHER": {
            "S": 0.75,
            "F": 0.25
        }
    },
    POSITIONS['EAST']: {
        "SHOOT": {
            "S": 0.9,
            "F": 0.1
        },
        "STAY": {
            "S": (1, POSITIONS['EAST']),
            "F": (0, POSITIONS['EAST'])
        },
        "LEFT": {
            "S": (1, POSITIONS['CENTER']),
            "F": (0, POSITIONS['EAST'])
        },
        "HIT": {
            "S": 0.2,"F": 0.8
        }
    },
    POSITIONS['WEST']: {
        "SHOOT": {
            "S": 0.25,
            "F": 0.75
        },
        "STAY": {
            "S": (1, POSITIONS['WEST']),
            "F": (0, POSITIONS['EAST'])
        },
        "RIGHT": {
            "S": (1, POSITIONS['CENTER']),
            "F": (0, POSITIONS['EAST'])
        },
    }
}

class State:
    def __init__(self, position, materials, arrows, mm_state, mm_health, act = 1):
        self.pos = position
        self.active = act
        self.materials = materials
        self.pos_mm = 0
        self.arrows = arrows
        self.mm_state = mm_state
        self.mm_health = mm_health

    def dec_mm_hp(self, val):
        self.mm_health -= val
        self.mm_health = min(self.mm_health, max_hp)
        self.mm_health = max(0, self.mm_health)

    def inc_mm_hp(self, val):
        self.mm_health += val
        self.mm_health = min(self.mm_health, max_hp)
        self.mm_health = max(0, self.mm_health)

    def set_arrows(self, arrows):
        self.arrows = max(0, arrows)
        self.arrows = min(max_num_arr, self.arrows)

    def get_state(self):
        if self.active == 1:
            return self.pos, self.materials, self.arrows, self.mm_state, self.mm_health
        return None

    def set_mats(self, count):
        self.materials = max(0, count)
        self.materials = min(max_num_mats, self.materials)


def get_all_actions(cur_state, V):
    state = State(*cur_state)
    # print(cur_state)
    # print("eee",V)

    actions_map = {}

    # GET BEST ACTION
    for chose_act, resp in ACTIONS[state.pos].items():
        next_actions = []

        state_nxt = State(*cur_state)
        # print("eee",chose_act)

        if chose_act in MOVE_CHOOSE:
            for _,(probability, next_position) in resp.items():
                # print('EEEYY',probability,next_position)
                state_nxt.pos = next_position
                val = state_nxt.get_state()
                if chose_act == "STAY":
                    next_actions.append((probability, val,1))
                else:
                    next_actions.append((probability, val,0))

        elif chose_act == "HIT":
            val = state_nxt.get_state()
            next_actions.append((resp['F'], val,0))
            state_nxt.dec_mm_hp(50)
            val = state_nxt.get_state()
            next_actions.append((resp['S'], val,0))

        elif chose_act == "SHOOT":
            if state.arrows != 0:
                state_nxt.set_arrows(state.arrows - 1)
                val = state_nxt.get_state()
                next_actions.append((resp['F'], val,0))
                state_nxt.dec_mm_hp(25)
                val = state_nxt.get_state()
                next_actions.append((resp['S'], val,0))
            else:
                continue

        elif chose_act == "GATHER":
            val = state_nxt.get_state()
            next_actions.append((resp['F'], val,0))
            state_nxt.set_mats(state_nxt.materials + 1)
            val = state_nxt.get_state()
            next_actions.append((resp['S'], val,0))

        elif chose_act == "CRAFT":
            if state.materials != 0:
                state_nxt.set_mats(state_nxt.materials - 1)
                for idx,prob in enumerate(resp):
                    # print(idx+1)
                    state_nxt.set_arrows(state_nxt.arrows+idx+1)
                    val = state_nxt.get_state()
                    next_actions.append((prob, val,0))
                    state_nxt.set_arrows(state_nxt.arrows-idx-1)
            else:
                continue

        actions_map[chose_act] = next_actions
    if actions_map:
        return actions_map
        
    return None

def mm_action(cur_state, V, actions_map):
    best_act_responses ,best_act = None,None
    # best_act = None 
    max_value = np.NINF
    for action, responses in actions_map.items():
        state = State(*cur_state)
        # print('action:',action)

        final_responses = []
        value = 0

        # IF STATE IS READY
        if state.mm_state == 1:
            for p, state_nxt,vv in responses:
                state_nxt = State(*state_nxt)
                final_responses.append((p * ready["STAY"], state_nxt.get_state(),  V[state_nxt.get_state()],0,vv))
                
                if state.pos != POSITIONS["EAST"] and state.pos != POSITIONS["CENTER"]:
                    state_nxt.mm_state = STATES_MM['DORMANT']
                    final_responses.append((p * ready["ATTACK"], state_nxt.get_state(),  V[state_nxt.get_state()],1,vv))

            if state.pos == POSITIONS["EAST"] or state.pos == POSITIONS["CENTER"]:
                state.set_arrows(0)
                state.inc_mm_hp(25)
                state.mm_state = STATES_MM['DORMANT']
                final_responses.append((ready["ATTACK"], state.get_state(),  V[state.get_state()],2,vv))
        
        # IF STATE IS DORMANT
        elif state.mm_state == 0:
            for p, state_nxt,vv in responses:
                state_nxt = State(*state_nxt)
                val = state_nxt.get_state()
                final_responses.append((p * dormant["STAY"], val, V[val],0,vv))
                state_nxt.mm_state = STATES_MM['READY']
                val = state_nxt.get_state()
                final_responses.append((p * dormant["GET_READY"], val, V[val],0,vv))

        for (prob, state, val, hitmarker,st) in final_responses:
            # if st == 1:
            #     cost=0
            # else:
            cost = STEP_COST
            if hitmarker != 2:
                value += ( prob * (cost + STATE_REWARD[state] + DISCOUNT * (V[state])))
            else:
                value += (prob * (MM_HIT_R + cost + STATE_REWARD[state] + DISCOUNT * (V[state])))

        # print('value',value)

        if value - max_value >= 0:
            max_value = value
            best_act_responses = final_responses
            best_act = action

    return best_act, best_act_responses, max_value

INIT_STATE = ((POSITIONS["WEST"],0,0,STATES_MM['DORMANT'],100))


if __name__=="__main__":
    
    start = INIT_STATE
    cnt = 0
    V = np.zeros((5, 3, 4, 2, 101))
    sim=True
    state_bact = {}
    state_bact_responses = {}
    over = True
    while over:
        new_V = np.zeros_like(V)
        max_delta = np.NINF
        prt = "iteration={}".format(cnt)
        print(prt)
        for state, _ in np.ndenumerate(V):
            # vals = "("+IJ_POS[state[0]+state[1]+state[2]+state[3]+state[4]+"):"
            if state[-1] in HEALTH_VVALS:
                if state[-1] != 0:
                    actions_map = get_all_actions(state, V)
                    if actions_map is None:
                        over=False
                    best_act, best_act_responses, value = mm_action(state, V, actions_map)
                    new_V[state] = value
                    # print('value',value)
                    ba = best_act
                    state_bact[state] = ba

                    state_bact_responses[state] = best_act_responses
                    if max_delta < abs(value - V[state]):
                        max_delta = abs(value - V[state])

                    print(f"({IJ_POS[state[0]]},{state[1]},{state[2]},{MM_STR[state[3]]},{state[4]}):{best_act}="  + "[{:0.3f}]".format(new_V[state]))
                    continue
                else:
                    # print(vals,"NONE=",new_V[state])
                    # print("(",IJ_POS[state[0]],state[1],state[2],MM_STR[state[3]],state[4],"):NONE=","[{:0.3f}]".format(new_V[state]))
                    print(f"({IJ_POS[state[0]]},{state[1]},{state[2]},{MM_STR[state[3]]},{state[4]}):NONE=" + "[{:0.3f}]".format(new_V[state]))
                    continue
                
            else:
                continue
        f = open("deltatrack.txt","a")
        prt = "iter: {}, delta: {}\n".format(cnt,max_delta)
        f.write(prt)
        # print(cnt, max_delta, file=)
        V = copy.deepcopy(new_V)
        f.close()
        # CONVERGENCE
        if max_delta < DELTA:
            over = False
        cnt += 1