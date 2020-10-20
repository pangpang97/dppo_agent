from itertools import combinations, product, combinations_with_replacement
import numpy as np
from configs import *
from mpsen_game_request import MpsenGameRequest
import time, json, random
import logging
import os
import traceback

logger = logging.getLogger(__name__)


class EnvironmentMpsen:
    def __init__(self):

        # 爬塔开启关卡
        self.__tower_start_stage = '2_13'
        # 快速挂机开启关卡
        self.__quick_reward_start_stage = '2_1'
        # 抽卡开启关卡
        self.__draw_card_start_stage = '2_1'
        # 退级开启关卡
        self.__level_exchange_start_stage = '2_1'
        # 7日礼包开启关卡
        self.__7day_reward_start_stage = '1_4'
        # 强者之路开启关卡
        self.__road_to_strong_start_stage = '2_5'
        # 公会开启关卡
        self.__guild_start_stage = '2_21'
        # 活跃通行证开启天数
        self.__master_objective_start_day = 8
        # 强者之路通行证开启关卡
        self.__master_road_start_stage = '7_1'

        # 请求尝试
        self.__retry_limit = 5
        self.__relogin_limit = 2

        # 结束关卡，通关后结束
        self.__done_next_stage = '4_29'

        # 卡关领取奖励
        self.__stage_stuck_limit = 100

        # 游戏天数
        self.__play_day_limit = 14

        # 领取挂机奖励次数，每次12小时奖励
        self.__quick_reward_limit = 2 * self.__play_day_limit

        # 史诗英雄
        self.__hero_template_epic = [3, 5, 9, 14, 22, 30, 47, 52, 53, 56, 57, 80]

        # 神话英雄
        self.__hero_template_mythical = [1, 2, 8, 10, 11, 12, 13, 18, 19, 20, 21, 23, 24, 25, 26, 27, 31, 32, 34, 35,
                                         36, 38, 39, 41, 43, 44, 45, 46, 49, 51, 61, 63, 84]
        # 全部英雄
        self.__hero_template_all = [3, 5, 9, 14, 22, 30, 47, 52, 53, 56, 57, 80, 1, 2, 8, 10, 11, 12, 13, 18, 19, 20,
                                    21, 23, 24, 25, 26, 27, 31, 32, 34, 35, 36, 38, 39, 41, 43, 44, 45, 46, 49, 51, 61,
                                    63, 84]

        # 位置列表
        self.__pos_list = list(range(0, 9))

        # action space
        ### 全部
        self.__action_space = []
        ### 有效的
        self.__action_space_legal = []
        ### 上阵英雄
        self.__action_space_choose_hero = []
        ### 交换场上位置action space
        self.__action_space_exchange_pos = []
        ### 英雄升级
        self.__action_space_hero_level_up = []
        ### 英雄进阶
        self.__action_space_hero_upgrade_quality = []
        ### 英雄等级交换
        self.__action_space_hero_level_exchange = []
        ### 装备强化
        self.__action_space_equip_upgrade_quality = []

        ### 上阵英雄
        self.__len_choose_hero = 0
        ### 交换场上位置action space
        self.__len_exchange_pos = 0
        ### 英雄升级
        self.__len_hero_level_up = 0
        ### 英雄进阶
        self.__len_hero_upgrade_quality = 0
        ### 英雄等级交换
        self.__len_hero_level_exchange = 0
        ### 装备强化
        self.__len_equip_upgrade_quality = 0

        # game role
        self.__oasis_id = None

        # game api request
        self.__game_request = None

        # game info
        self.__init_game_info()

        # action space
        self.__init_action_space()

    def __init_game_info(self):
        # 错误次数
        self.__except_count = 0

        # 领取快速挂机奖励次数，每次12小时奖励
        self.__quick_reward_count = 0

        # 卡关次数，同一关卡，失败次数
        self.__stage_stuck_count = 0

        ### 玩家游戏信息
        # 进度
        self.__next_stage = '1_1'

        # 爬塔
        self.__tower_floor = 0

        # 钻石
        self.__diamond = 0

        # 英雄经验
        self.__hero_exp = 0

        # 装备经验
        self.__equip_exp_10_104 = 0
        self.__equip_exp_1000_105 = 0

        # 代币
        self.__coin = 0

        # 英雄升级紫钻
        self.__hero_powder = 0

        # 活跃经验
        self.__master_objective_exp = 0
        self.__master_objective_level = 0

        # 强者之路经验
        self.__master_road_exp = 0
        self.__master_road_level = 0

        # {英雄id：[14个长度的list，每个对应相应品阶的个数，没有为0]}
        self.__hero_quality_count = {}
        for hero in self.__hero_template_all:
            self.__hero_quality_count[hero] = [0] * 14

        # 玩家拥有角色情况
        self.__hero_list = {}

        # 上阵阵容，英雄模板id-英雄实例id
        self.__battle_lineup = []

        # 普通招募令数量
        self.__recruit_token_amount = 0

        # 类型招募令数量
        self.__type_recruit_token_amount = 0

        # 精英招募令数量
        self.__elite_recruit_token_amount = 0

        # 类型招募开启类别
        self.__type_draw_card_opened = []

        # team level 现有卡牌中，等级排第5的值
        self.__team_level = 1

        # 玩家拥有角色template统计
        self.__owned_hero_template = {}

        # 玩家拥有角色instance统计
        self.__owned_hero_instance = {}

        # 玩家拥有装备template统计, {type: {role: equip-list}}
        self.__owned_equip_template = {}

        # 玩家拥有装备instance统计
        self.__owned_equip_instance = {}

        # 已领取了哪天的礼包
        self.__seven_day_rewarded = []

        # 游戏天数
        self.__played_days = 0

        # 竞技场战斗次数
        self.__arena_count = 0

        # 对战信息
        self.__lineup_force = 0
        self.__lineup_force_before = 0
        self.__lineup_damage = 0
        self.__defence_alive = 0
        self.__defence_force = 0

    def get_observation_length(self):
        return len(self.__get_observation())

    def get_action_space_length(self):
        return len(self.__action_space)

    # 重置初始化
    def reset(self):
        self.__init_game_info()

        self.__oasis_id = str(os.getpid()) + str(int(time.time()))
        # self.__oasis_id = 151061601002645

        if self.__game_request is not None:
            self.__game_request.get_sock_clt().close()
        self.__game_request = MpsenGameRequest(self.__oasis_id)
        self.__set_user_base_data()

        self.__get_once_reward()

        self.__battle_lineup = {'template': [0] * 9, 'instance': [0] * 9}

        self.__update_legal_action_space()

        return self.__get_observation()

    def get_action_space(self):
        return list(range(0, len(self.__action_space)))

    def get_legal_action_space(self):
        return self.__action_space_legal

    # observation
    # ob，总长度32: 上阵英雄（9个位置）、品质（9个位置）、等级（9个位置）、关卡id、上次对方存活人数、战力差距（我-敌）、战力差距（本次-上次）、我方伤害
    def __get_observation(self):
        ob = []

        # 上阵英雄（9个位置）、品质（9个位置）、等级（9个位置）
        lineup_hero = []
        lineup_quality = []
        lineup_level = []
        for hero in self.__battle_lineup['instance']:
            if hero:
                lineup_hero.append(self.__owned_hero_instance[hero]['heroId'])
                lineup_quality.append(self.__owned_hero_instance[hero]['quality'])
                lineup_level.append(self.__owned_hero_instance[hero]['level'])
            else:
                lineup_hero.append(0)
                lineup_quality.append(0)
                lineup_level.append(0)
        ob.extend(lineup_hero)
        ob.extend(lineup_quality)
        ob.extend(lineup_level)

        # 关卡id、上次对方存活人数、战力差距（我-敌）、战力差距（本次-上次）、我方伤害
        ob.extend([
            STAGE_ID_INFO[self.__next_stage],
            self.__defence_alive,
            self.__lineup_force - self.__defence_force,
            self.__lineup_force - self.__lineup_force_before,
            self.__lineup_damage
        ])

        # handle invalid value
        ob = np.array(ob)
        ob[np.isnan(ob)] = 0
        ob[np.isinf(ob)] = 0

        return list(ob)

    # 设置用户资源
    def __set_user_base_data(self):
        # reset game resource
        self.__hero_quality_count = {}
        for hero in self.__hero_template_all:
            self.__hero_quality_count[hero] = [0] * 14
        self.__hero_list = {}
        self.__owned_hero_template = {}
        self.__owned_hero_instance = {}
        self.__owned_equip_template = {}
        self.__owned_equip_instance = {}

        user_info = self.__game_request.user_base_data()
        # user hero
        if 'userHeroes' in user_info:
            for tmp_hero in user_info['userHeroes']:
                if tmp_hero['heroId'] not in self.__owned_hero_template:
                    self.__owned_hero_template[tmp_hero['heroId']] = []

                self.__owned_hero_template[tmp_hero['heroId']].append(tmp_hero)
                self.__owned_hero_instance[tmp_hero['id']] = tmp_hero

        for tmp_i in self.__owned_hero_template:
            self.__owned_hero_template[tmp_i].sort(key=lambda x: (-x['level'], -x['quality']))

        # user equip
        if 'userEquips' in user_info:
            for tmp_equip in user_info['userEquips']:
                tmp_equip['type'] = EQUIPMENT_LIST[tmp_equip['equipId']][0]
                tmp_equip['role'] = EQUIPMENT_LIST[tmp_equip['equipId']][1]
                tmp_equip['quality'] = EQUIPMENT_LIST[tmp_equip['equipId']][2]

                if tmp_equip['type'] not in self.__owned_equip_template:
                    self.__owned_equip_template[tmp_equip['type']] = {}
                if tmp_equip['role'] not in self.__owned_equip_template[tmp_equip['type']]:
                    self.__owned_equip_template[tmp_equip['type']][tmp_equip['role']] = []

                self.__owned_equip_template[tmp_equip['type']][tmp_equip['role']].append(tmp_equip)
                self.__owned_equip_instance[tmp_equip['id']] = tmp_equip

        for tmp_type in self.__owned_equip_template:
            for tmp_role in self.__owned_equip_template[tmp_type]:
                self.__owned_equip_template[tmp_type][tmp_role].sort(key=lambda x: (-x['quality']))

        user_asset = user_info.get('userAsset', [])
        user_props = user_info.get('userProps', [])
        quality_position_map = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12,
                                16: 13}
        position_quality_map = {v: k for k, v in quality_position_map.items()}
        user_heros = user_info.get('userHeroes', [])
        hero_level_map = {}
        hero_level_arr = []
        for hero in user_heros:
            hero_id = int(hero['heroId'])
            hero_level = int(hero['level'])
            hero_level_arr.append(hero_level)
            if hero_level_map.get(hero_id, None) is None:
                hero_level_map[hero_id] = hero_level
            else:
                curr_level = hero_level_map[hero_id]
                if curr_level < hero_level:
                    hero_level_map[hero_id] = hero_level
        if len(hero_level_arr) >= 5:
            self.__team_level = sorted(hero_level_arr, reverse=True)[4]
        for hero in user_heros:
            quality = int(hero['quality'])
            hero_id = int(hero['heroId'])
            hero_key_id = int(hero['id'])
            hero_equips = hero.get('equips', {})
            if quality < 3:
                continue
            key = quality_position_map[quality]
            if self.__hero_quality_count.get(hero_id, None) is None:
                self.__hero_quality_count[hero_id] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.__hero_quality_count[hero_id][key] = self.__hero_quality_count[hero_id][key] + quality
            else:
                self.__hero_quality_count[hero_id][key] = self.__hero_quality_count[hero_id][key] + quality
            if self.__hero_list.get(hero_id, None) is None:
                self.__hero_list[hero_id] = {'equips': [], 'max_level': 1}
                self.__hero_list[hero_id]['equips'] = {hero_key_id: hero_equips}
            else:
                self.__hero_list[hero_id]['equips'][hero_key_id] = hero_equips
            self.__hero_list[hero_id]['max_level'] = hero_level_map[hero_id]
        for hero_id in self.__hero_quality_count:
            for key, quality_id in enumerate(self.__hero_quality_count.get(hero_id, [])):
                self.__hero_quality_count[hero_id][key] = int(
                    self.__hero_quality_count[hero_id][key] / position_quality_map[key])
        for props in user_props:
            if int(props['specId']) == 104:
                self.__equip_exp_10_104 = props.get('amount', 0)
            if int(props['specId']) == 105:
                self.__equip_exp_1000_105 = props.get('amount', 0)
            if int(props['specId']) == 106:
                self.__recruit_token_amount = props.get('amount', 0)
            if int(props['specId']) == 107:
                self.__type_recruit_token_amount = props.get('amount', 0)

        self.__coin = int(user_asset.get('coin', 0))
        self.__diamond = int(user_asset.get('diamond', 0))
        self.__hero_exp = int(user_asset.get('heroExp', 0))
        self.__hero_powder = int(user_asset.get('heroPowder', 0))
        self.__next_stage = self.__game_request.next_stage_id
        self.__elite_recruit_token_amount = int(user_asset.get('typeDrawCardNum', 0))
        self.__type_draw_card_opened = user_asset.get('roleDrawCardOpened', [])
        self.__lineup_force = user_info.get("stage_team_force", 0)

    """
    @desc 查看是否有闯关奖励
    @return boolean True:有,False:无
    """

    def __check_mission_reward(self):
        return self.__game_request.is_mission_ready()

    # 执行action
    def step(self, action_index):
        reward = None
        done = False

        try:
            action = self.__action_space[action_index]
            logger.debug("action index: {}, action: {}".format(action_index, action))
            if action_index < self.__len_choose_hero:
                hero = action[0]
                pos = action[1]

                self.__battle_lineup['template'][pos] = hero
                self.__battle_lineup['instance'][pos] = self.__owned_hero_template[hero][0]['id']
            elif action_index < self.__len_exchange_pos:
                pos1 = action[0]
                pos2 = action[1]

                hero1 = self.__battle_lineup['template'][pos1]
                hero2 = self.__battle_lineup['template'][pos2]
                self.__battle_lineup['template'][pos1] = hero2
                self.__battle_lineup['template'][pos2] = hero1

                hero1 = self.__battle_lineup['instance'][pos1]
                hero2 = self.__battle_lineup['instance'][pos2]
                self.__battle_lineup['instance'][pos1] = hero2
                self.__battle_lineup['instance'][pos2] = hero1
            elif action_index < self.__len_hero_level_up:
                self.__game_request.hero_level_up(int(self.__owned_hero_template[action][0]['id']),
                                                  self.__owned_hero_template[action][0]['level'] + 1)
            elif action_index < self.__len_hero_upgrade_quality:
                hero = action[0]
                quality = action[1]
                cost1 = action[2]
                cost2 = action[3]

                upgrade_id = 0
                cost_id = []
                cost_quality = HERO_QUALITY[quality][2]

                for i in self.__owned_hero_template[hero]:
                    if i['quality'] == quality:
                        upgrade_id = i['id']
                        break

                for i in self.__owned_hero_template[cost1]:
                    if i['quality'] == cost_quality and i['id'] != upgrade_id:
                        cost_id.append(i['id'])
                        break

                if cost2:
                    for i in self.__owned_hero_template[cost2]:
                        if i['quality'] == cost_quality and i['id'] != upgrade_id and i['id'] not in cost_id:
                            cost_id.append(i['id'])
                            break

                self.__game_request.hero_compose(upgrade_id, cost_id)
            elif action_index < self.__len_hero_level_exchange:
                hero1 = action[0]
                hero2 = action[1]
                hero1_level = self.__owned_hero_template[hero1][0]['level']
                hero2_level = self.__owned_hero_template[hero2][0]['level']

                # 交换等级
                if hero1_level > hero2_level:
                    self.__game_request.hero_level_down(int(self.__owned_hero_template[hero1][0]['id']), hero2_level)
                    self.__game_request.hero_level_up(int(self.__owned_hero_template[hero2][0]['id']), hero1_level)
                else:
                    self.__game_request.hero_level_down(int(self.__owned_hero_template[hero2][0]['id']), hero1_level)
                    self.__game_request.hero_level_up(int(self.__owned_hero_template[hero1][0]['id']), hero2_level)

                # 修改阵容
                if hero1 in self.__battle_lineup['template'] and hero2 not in self.__battle_lineup['template']:
                    lineup_index = self.__battle_lineup['template'].index(hero1)
                    self.__battle_lineup['template'][lineup_index] = hero2
                    self.__battle_lineup['instance'][lineup_index] = self.__owned_hero_template[hero2][0]['id']
                elif hero2 in self.__battle_lineup['template'] and hero1 not in self.__battle_lineup['template']:
                    lineup_index = self.__battle_lineup['template'].index(hero2)
                    self.__battle_lineup['template'][lineup_index] = hero1
                    self.__battle_lineup['instance'][lineup_index] = self.__owned_hero_template[hero1][0]['id']
            elif action_index < self.__len_equip_upgrade_quality:
                hero = action[0]
                type = action[1]

                lineup_index = self.__battle_lineup['template'].index(hero)
                hero_instance = self.__owned_hero_instance[self.__battle_lineup['instance'][lineup_index]]

                self.__game_request.equip_enhance(hero_instance['equips'][type], [],
                                                  {104: self.__equip_exp_10_104, 105: self.__equip_exp_1000_105})

            # combat
            tmp_reward = 0
            total_reward = 0

            # 自动穿戴装备
            self.__auto_equip()

            # 战斗阵容
            combat_dic = {}
            for i in self.__battle_lineup['instance']:
                if i:
                    index = self.__battle_lineup['instance'].index(i)
                    combat_dic[index + 1] = int(i)

            # 胜利后继续下一关
            while True:
                combat_rs = self.__game_request.combat(combat_dic, 1)
                if combat_rs:
                    tmp_reward = self.__calc_reward(combat_rs)
                    if tmp_reward > 0:
                        total_reward += tmp_reward
                        self.__set_user_base_data()
                    else:
                        break

            # 胜利
            if total_reward > 0:
                reward = total_reward

                # 卡关次数清零
                self.__stage_stuck_count = 0

                # 爬塔
                if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__tower_start_stage]:
                    try:
                        while True:
                            tmp_tower_rs = self.__game_request.combat(combat_dic, 2)
                            if not tmp_tower_rs.get('userSimu'):
                                break

                            if tmp_tower_rs.get('userSimu').get('floor') <= self.__tower_floor:
                                break

                            self.__tower_floor = tmp_tower_rs.get('userSimu').get('floor')
                    except Exception as e:
                        print("fight tower failed")
                        traceback.print_exc()
            else:
                reward = tmp_reward

                # 卡关次数+1
                self.__stage_stuck_count += 1
                if self.__stage_stuck_count % self.__stage_stuck_limit == 0 \
                        and self.__quick_reward_count < self.__quick_reward_limit \
                        and STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__quick_reward_start_stage]:
                    self.__game_request.get_resource(0, 2)
                    self.__game_request.get_resource(0, 2)
                    self.__game_request.get_resource(0, 2)
                    self.__game_request.get_resource(0, 2)
                    self.__game_request.get_resource(0, 2)
                    self.__game_request.get_resource(0, 2)

                    self.__quick_reward_count += 1

            # 7日礼包
            self.__get_7day_reward()

            # 玩法奖励
            self.__get_reward()

            # 里程碑奖励
            self.__game_request.get_mission_reward()

            # 道具使用
            self.__game_request.use_props()

            # 抽卡
            self.__set_user_base_data()
            if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__draw_card_start_stage]:
                # 普通抽：道具
                for i in range(self.__recruit_token_amount):
                    self.__game_request.draw_card(2, 1)

                # 普通抽：钻石
                if self.__diamond >= DRAW_CARD_COST_DIAMOND_10:
                    self.__game_request.draw_card(2, 2)

                # 类型抽
                if self.__type_recruit_token_amount:
                    if 5 not in self.__type_draw_card_opened:
                        self.__game_request.put_resource_to_user("diamond,300")
                        self.__game_request.draw_card_open(5)
                    for i in range(self.__type_recruit_token_amount):
                        self.__game_request.draw_card(5, 1)

                # 精英抽
                for i in range(self.__elite_recruit_token_amount):
                    self.__game_request.draw_card_type(2)

            # 异常次数清零
            self.__except_count = 0
        except Exception as e:
            traceback.print_exc()

            # 异常次数+1，重新登录
            self.__except_count += 1
            if self.__except_count >= 10:
                if self.__game_request is not None:
                    self.__game_request.get_sock_clt().close()
                self.__game_request = MpsenGameRequest(self.__oasis_id)
                self.__except_count = 0

        # 任何action执行完，更新env info
        self.__set_user_base_data()
        self.__update_legal_action_space()

        if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__done_next_stage]:
            done = True
            reward = 100000

        if reward is None or not np.isfinite(reward):
            reward = 0

        ob = self.__get_observation()

        self.__lineup_force_before = self.__lineup_force

        return ob, reward, done, {}

    # 计算reward
    def __calc_reward(self, combat_rs):
        if combat_rs.get('isWin') == True:
            if self.__next_stage.split('_')[0] != '1' and self.__next_stage.split('_')[1] == '1':
                reward = 10 * (1 + int(self.__next_stage.split('_')[0]) / int(self.__done_next_stage.split('_')[0]))
            else:
                reward = (1 * (1 + STAGE_ID_INFO[self.__next_stage] / STAGE_ID_INFO[
                    self.__done_next_stage])) / combat_rs.get("fightRound", 1)
        else:
            defender_hp_remainning = sum(combat_rs['defenderHps'].values())
            defender_hp_total = defender_hp_remainning + sum(
                int(item.get('undertake', 0)) - int(item.get('heal', 0)) for item in combat_rs['defenderStats'].values()
            )

            reward = -defender_hp_remainning / defender_hp_total

            self.__lineup_damage = int(combat_rs.get('damage', 0))
            self.__defence_alive = sum(1 for i in combat_rs['defenderHps'] if combat_rs['defenderHps'][i] > 0)
            self.__defence_force = self.__game_request.get_bot_team_info(STAGE_ID_INFO[self.__next_stage])['team_force']

        return reward

    # 更新有效action space
    def __update_legal_action_space(self):
        self.__action_space_legal = []
        for action_index in range(len(self.__action_space)):
            action = self.__action_space[action_index]
            if action_index < self.__len_choose_hero:
                hero = action[0]
                pos = action[1]

                if not self.__owned_hero_template.get(hero) \
                        or hero in self.__battle_lineup['template'] \
                        or (np.count_nonzero(self.__battle_lineup['template']) == 5 and not
                self.__battle_lineup['template'][pos]):
                    continue
            elif action_index < self.__len_exchange_pos:
                continue
                pos1 = action[0]
                pos2 = action[1]
                hero1 = self.__battle_lineup['template'][pos1]
                hero2 = self.__battle_lineup['template'][pos2]

                if not hero1 and not hero2:
                    continue
            elif action_index < self.__len_hero_level_up:
                hero = action
                if not self.__owned_hero_template.get(hero):
                    continue
                elif hero not in self.__battle_lineup['template']:
                    continue
                elif not self.__check_level_up(self.__owned_hero_template[hero][0]['quality'],
                                               self.__owned_hero_template[hero][0]['level'],
                                               self.__owned_hero_template[hero][0]['level'] + 10):
                    continue
            elif action_index < self.__len_hero_upgrade_quality:
                if not self.__check_quality_level_up(action[0], action[1], action[2], action[3]):
                    continue
            elif action_index < self.__len_hero_level_exchange:
                hero1 = action[0]
                hero2 = action[1]

                if STAGE_ID_INFO[self.__next_stage] < STAGE_ID_INFO[self.__level_exchange_start_stage] \
                        or self.__diamond < 20 \
                        or not self.__owned_hero_template.get(hero1) \
                        or not self.__owned_hero_template.get(hero2) \
                        or self.__owned_hero_template[hero1][0]['level'] == self.__owned_hero_template[hero2][0][
                    'level'] \
                        or (self.__owned_hero_template[hero1][0]['level'] <= 1 and self.__owned_hero_template[hero2][0][
                    'level'] <= 1):
                    continue
            elif action_index < self.__len_equip_upgrade_quality:
                hero = action[0]
                type = action[1]

                # 没有英雄
                if not self.__owned_hero_template.get(hero):
                    continue
                # 英雄没上阵
                elif hero not in self.__battle_lineup['template']:
                    continue
                else:
                    lineup_index = self.__battle_lineup['template'].index(hero)
                    hero_instance_id = self.__battle_lineup['instance'][lineup_index]
                    # 英雄没穿装备
                    if 'equips' not in self.__owned_hero_instance[hero_instance_id]:
                        continue
                    # type位置没穿装备
                    elif type not in self.__owned_hero_instance[hero_instance_id]['equips']:
                        continue
                    else:
                        equip_id = self.__owned_hero_instance[hero_instance_id]['equips'][type]
                        equip_instance = self.__owned_equip_instance[equip_id]
                        quality = equip_instance['quality']
                        level = equip_instance.get('enhanceLevel', 0)
                        exp = equip_instance.get('enhanceExp', 0)
                        if not self.__check_equipment_level_up(quality, level, exp):
                            continue
            else:
                continue

            self.__action_space_legal.append(action_index)

    # 上阵英雄，同一个英雄模板，只上阵品阶、等级最高的
    def __init_action_space_choose_hero(self):
        self.__action_space_choose_hero = list(product(self.__hero_template_all, self.__pos_list))

    # 交换场上位置动作空间
    def __init_action_space_exchange_pos(self):
        self.__action_space_exchange_pos = list(combinations(self.__pos_list, 2))

    # 升级英雄，同一个英雄模板，只升级品阶、等级最高的
    def __init_action_space_hero_level_up(self):
        self.__action_space_hero_level_up = self.__hero_template_all

    # 英雄合成
    def __init_action_space_hero_upgrade_quality(self):
        self.__action_space_hero_upgrade_quality = []
        for hero in self.__hero_template_all:
            if HERO_QUALITY_LEVEL_UP_CONDITIONS[hero][0] == HERO_ROLE_GRADING_EPIC:
                # rare，消耗2个rare自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_RARE, hero, hero])
                # elite，消耗1个elite自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_ELITE, hero, None])
                # epic，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_EPIC, hero, None])

                # rare+，消耗2个rare+同类
                for i in combinations_with_replacement(
                        HERO_TYPE_QUALITY_LIMIT[HERO_ROLE_GRADING_EPIC][HERO_QUALITY_LEVEL_UP_CONDITIONS[hero][1]], 2):
                    self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_RARE_PLUS, i[0], i[1]])

            elif HERO_QUALITY_LEVEL_UP_CONDITIONS[hero][0] == HERO_ROLE_GRADING_MYTHICAL:
                # elite，消耗1个elite自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_ELITE, hero, None])
                # epic，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_EPIC, hero, None])
                # lagendary+，消耗2个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_LAGENDARY_PLUS, hero, hero])
                # mythical，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_MYTHICAL, hero, None])
                # mythical+1，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_MYTHICAL_PLUS_ONE, hero, None])
                # mythical+2，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_MYTHICAL_PLUS_TWO, hero, None])
                # mythical+3，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_MYTHICAL_PLUS_THREE, hero, None])
                # mythical+4，消耗1个elite+自己
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_MYTHICAL_PLUS_FOUR, hero, None])

                for i in HERO_TYPE_QUALITY_LIMIT[HERO_ROLE_GRADING_MYTHICAL][HERO_QUALITY_LEVEL_UP_CONDITIONS[hero][1]]:
                    # epic+，消耗1个epic+同类
                    self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_EPIC_PLUS, i, None])
                    # lagendary，消耗1个epic+同类
                    self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_LAGENDARY, i, None])
            else:
                continue

            # elite+，消耗2个elite+同类
            for i in combinations_with_replacement(
                    HERO_TYPE_QUALITY_LIMIT[HERO_ROLE_GRADING_ALL][HERO_QUALITY_LEVEL_UP_CONDITIONS[hero][1]], 2):
                self.__action_space_hero_upgrade_quality.append([hero, HERO_QUALITY_ELITE_PLUS, i[0], i[1]])

    # 英雄等级互换，最少1个等级大于等于2级的
    def __init_action_space_hero_level_exchange(self):
        self.__action_space_hero_level_exchange = list(combinations(self.__hero_template_all, 2))

    # 装备强化，5个场上英雄，4个装备位置
    def __init_action_space_equip_upgrade_quality(self):
        self.__action_space_equip_upgrade_quality = list(product([1, 2, 3, 4, 5], [1, 2, 3, 4]))

    # action space 初始化
    def __init_action_space(self):
        # 选英雄上阵
        self.__init_action_space_choose_hero()
        self.__action_space.extend(self.__action_space_choose_hero)
        self.__len_choose_hero = len(self.__action_space)

        # 改变场上英雄站位
        self.__init_action_space_exchange_pos()
        self.__action_space.extend(self.__action_space_exchange_pos)
        self.__len_exchange_pos = len(self.__action_space)

        # 英雄升级
        self.__init_action_space_hero_level_up()
        self.__action_space.extend(self.__action_space_hero_level_up)
        self.__len_hero_level_up = len(self.__action_space)

        # 英雄品质提升
        self.__init_action_space_hero_upgrade_quality()
        self.__action_space.extend(self.__action_space_hero_upgrade_quality)
        self.__len_hero_upgrade_quality = len(self.__action_space)

        # 英雄等级互换
        self.__init_action_space_hero_level_exchange()
        self.__action_space.extend(self.__action_space_hero_level_exchange)
        self.__len_hero_level_exchange = len(self.__action_space)

        # 装备强化
        self.__init_action_space_equip_upgrade_quality()
        self.__action_space.extend(self.__action_space_equip_upgrade_quality)
        self.__len_equip_upgrade_quality = len(self.__action_space)

    """
    @desc 英雄升级
    @param int quality 品质 
    @param int level_from  初始英雄等级
    @param int level_to  最终英雄等级
    @return boolean 是否可以升级
    """

    def __check_level_up(self, quality, level_from, level_to):
        quality = int(quality)
        level_from = int(level_from)
        level_to = int(level_to)
        if level_from in HERO_LEVEL_UP_CONDITIONS.keys():
            conditions = HERO_LEVEL_UP_CONDITIONS[level_from]
        else:
            return False

        if level_to > HERO_QUALITY_LEVEL_LIMIT[quality]:
            return False

        if self.__hero_exp >= conditions[0] and self.__coin >= conditions[1] and self.__hero_powder >= conditions[
            2] and self.__team_level >= \
                conditions[
                    3]:
            return True
        else:
            return False

    """
    @desc 英雄品质提升 
    @param int hero_id 英雄id 
    @param int quality  当前品质
    @param int consume_hero_id_1  狗粮英雄id
    @param int consume_hero_id_2  狗粮英雄id
    @return boolean 是否可以提升
    """

    def __check_quality_level_up(self, hero_id, quality, consume_hero_id_1=None, consume_hero_id_2=None):

        # 0=index  1=升级消耗个数  2=消耗的品质
        cost_quality = HERO_QUALITY[quality][2]

        needs_list = {
            hero_id: {quality: 1}
        }

        if consume_hero_id_1:
            if consume_hero_id_1 not in needs_list:
                needs_list[consume_hero_id_1] = {cost_quality: 1}
            elif cost_quality not in needs_list[consume_hero_id_1]:
                needs_list[consume_hero_id_1][cost_quality] = 1
            else:
                needs_list[consume_hero_id_1][cost_quality] += 1

        if consume_hero_id_2:
            if consume_hero_id_2 not in needs_list:
                needs_list[consume_hero_id_2] = {cost_quality: 1}
            elif cost_quality not in needs_list[consume_hero_id_2]:
                needs_list[consume_hero_id_2][cost_quality] = 1
            else:
                needs_list[consume_hero_id_2][cost_quality] += 1

        # check
        for tmp_hero in needs_list:
            for tmp_quality in needs_list[tmp_hero]:
                count_index = HERO_QUALITY[tmp_quality][0]
                if needs_list[tmp_hero][tmp_quality] > self.__hero_quality_count[tmp_hero][count_index]:
                    return False

        return True

        # if hero_id in HERO_QUALITY_LEVEL_UP_CONDITIONS.keys():
        #     conditions = HERO_QUALITY_LEVEL_UP_CONDITIONS[hero_id]
        # else:
        #     return False
        #     # raise Exception("Invalid hero_id!", hero_id)
        #
        # role_level = conditions[0]
        # role_type = conditions[1]
        #
        # # 最高等级 无需在升级
        # if quality == HERO_QUALITY_MYTHICAL_PLUS_FIVE:
        #     return False
        #     # raise Exception("The hero quality is already the highest")
        #
        # # 角色分级无法突破史诗+
        # if quality == HERO_QUALITY_EPIC_PLUS and role_level != HERO_ROLE_GRADING_MYTHICAL:
        #     return False
        #     # raise Exception("The hero can't be upgrade quality")
        #
        # # get user hero list
        # # ['rare', 'rare+', 'elite', 'elite+', 'epic', 'epic+', 'lagendary',
        # # 'lagendary+', 'mythical','mythical+1', 'mythical+2', 'mythical+3', 'mythical+4¬', 'mythical+5']
        # hero_list = self.__hero_quality_count
        #
        # if hero_list[hero_id][HERO_QUALITY[quality][0]] == 0:
        #     return False
        #
        # upgrade_conditions = HERO_QUALITY[quality]
        # upgrade_num = upgrade_conditions[1]
        # upgrade_quality = upgrade_conditions[2]
        # upgrade_type = upgrade_conditions[3]
        #
        # # 升星要求俩个消耗英雄
        # if upgrade_num == 2:
        #     if consume_hero_id_1 != None and consume_hero_id_2 != None:
        #         # 升星消耗要求是自己
        #         if upgrade_type == 0:
        #             if consume_hero_id_1 == hero_id and consume_hero_id_2 == hero_id:
        #                 # 如果升级需求品质与当前英雄品质相同 并且升级消耗是本英雄的情况 需求+1
        #                 if upgrade_quality == quality:
        #                     remaining_quantity_required = upgrade_num + 1
        #                 else:
        #                     remaining_quantity_required = upgrade_num
        #                 # 检查自己是否有品质符合要求，可供升级消耗的英雄
        #                 if hero_list[hero_id][HERO_QUALITY[upgrade_quality][0]] < remaining_quantity_required:
        #                     return False
        #                     # raise Exception("Lack of consumables for hero upgrade by self")
        #             else:
        #                 return False
        #                 # raise Exception("The hero upgrade need two self!")
        #         else:
        #             consume_type_1 = HERO_QUALITY_LEVEL_UP_CONDITIONS[consume_hero_id_1][1]
        #             consume_type_2 = HERO_QUALITY_LEVEL_UP_CONDITIONS[consume_hero_id_2][1]
        #             # 升星消耗要求是同类英雄
        #             if consume_type_1 == role_type and consume_type_2 == role_type:
        #                 # check参数升级消耗英雄是否等于自己
        #                 if consume_hero_id_1 == hero_id and consume_hero_id_1 == hero_id:
        #                     # 如果升级需求品质与当前英雄品质相同 并且升级消耗是本英雄的情况 需求+1
        #                     if upgrade_quality == quality:
        #                         remaining_quantity_required = upgrade_num + 1
        #                     else:
        #                         remaining_quantity_required = upgrade_num
        #
        #                     if hero_list[consume_hero_id_1][
        #                         HERO_QUALITY[upgrade_quality][0]] < remaining_quantity_required:
        #                         return False
        #                         # raise Exception("Lack of consumables for hero upgrade! (consume_hero = hero_id)")
        #                 # check consume_hero_id_1 与本英雄相同
        #                 elif consume_hero_id_1 == hero_id:
        #                     if hero_list[consume_hero_id_1][HERO_QUALITY[upgrade_quality][0]] < upgrade_num or \
        #                             hero_list[consume_hero_id_2][HERO_QUALITY[upgrade_quality][0]] == 0:
        #                         return False
        #                         # raise Exception("Lack of consumables for hero upgrade  by consume_hero_id_1!")
        #                 # check consume_hero_id_2 与本英雄相同
        #                 elif consume_hero_id_2 == hero_id:
        #                     if hero_list[consume_hero_id_1][HERO_QUALITY[upgrade_quality][0]] == 0 or \
        #                             hero_list[consume_hero_id_2][HERO_QUALITY[upgrade_quality][0]] < upgrade_num:
        #                         return False
        #                         # raise Exception("Lack of consumables for hero upgrade by consume_hero_id_2!")
        #                 # check  升级消耗参数都不等于本英雄
        #                 else:
        #                     if hero_list[consume_hero_id_1][HERO_QUALITY[upgrade_quality][0]] == 0 or \
        #                             hero_list[consume_hero_id_2][HERO_QUALITY[upgrade_quality][0]] == 0:
        #                         return False
        #                         # raise Exception("Lack of consumables for hero upgrade!")
        #             else:
        #                 return False
        #                 # raise Exception("The hero upgrade need two same type hero!")
        #     else:
        #         return False
        #         # raise Exception("The hero upgrade need two  consume hero!")
        # else:
        #     # 升星要求一个消耗英雄
        #     if consume_hero_id_1 != None:
        #         # 升星消耗要求是自己
        #         if upgrade_type == 0:
        #             if consume_hero_id_1 == hero_id:
        #                 # 如果升级需求品质与当前英雄品质相同 并且升级消耗是本英雄的情况 需求+1
        #                 if upgrade_quality == quality:
        #                     remaining_quantity_required = upgrade_num + 1
        #                 else:
        #                     remaining_quantity_required = upgrade_num
        #                 # 检查自己是否有品质符合要求，可供升级消耗的英雄
        #                 if hero_list[hero_id][HERO_QUALITY[upgrade_quality][0]] < remaining_quantity_required:
        #                     return False
        #                     # raise Exception("Lack of one consumables for hero upgrade by self")
        #             else:
        #                 return False
        #                 # raise Exception("The hero upgrade need one self! ")
        #         else:
        #             consume_type_1 = HERO_QUALITY_LEVEL_UP_CONDITIONS[consume_hero_id_1][1]
        #             # 升星消耗要求是同类英雄
        #             if consume_type_1 == role_type:
        #                 # check参数升级消耗英雄是否等于自己
        #                 if consume_hero_id_1 == hero_id:
        #                     # 如果升级需求品质与当前英雄品质相同 并且升级消耗是本英雄的情况 需求+1
        #                     if upgrade_quality == quality:
        #                         remaining_quantity_required = upgrade_num + 1
        #                     else:
        #                         remaining_quantity_required = upgrade_num
        #
        #                     if hero_list[consume_hero_id_1][
        #                         HERO_QUALITY[upgrade_quality][0]] < remaining_quantity_required:
        #                         return False
        #                         # raise Exception("Lack of one consumables for hero upgrade! (consume_hero = hero_id)")
        #                 # check consume_hero_id_1 与本英雄相同
        #                 elif consume_hero_id_1 == hero_id:
        #                     if hero_list[consume_hero_id_1][HERO_QUALITY[upgrade_quality][0]] < upgrade_num:
        #                         return False
        #                         # raise Exception("Lack of one consumables for hero upgrade")
        #                 # check  升级消耗参数都不等于本英雄
        #                 else:
        #                     if hero_list[consume_hero_id_1][HERO_QUALITY[upgrade_quality][0]] == 0:
        #                         return False
        #                         # raise Exception("Lack of one consumables for hero upgrade!")
        #             else:
        #                 return False
        #                 # raise Exception("The hero upgrade need one same type hero!")
        #
        #     else:
        #         return False
        #         # raise Exception("The hero upgrade need one  consume hero!")
        #
        # return True

    """
    @desc 装备强化 
    @param int quality  当前品质
    @param int level  当前装备等级
    @param int current_exp  当前装备拥有经验
    @return boolean 是否可以强化
    """

    def __check_equipment_level_up(self, quality, level, current_exp):
        if quality in EQUIPMENT_LEVEL_UP_CONDITIONS.keys():
            quality_limit = EQUIPMENT_LEVEL_UP_CONDITIONS[quality]
        else:
            return False
            # raise Exception("Invalid quality!", quality)

        if level >= len(quality_limit):
            return False
            # raise Exception("Invalid level!", level)

        conditions = quality_limit[level]

        if level >= conditions[1]:
            return False
            # raise Exception("The level is already the highest!", level)

        total_exp = self.__equip_exp_10_104 * EQUIPMENT_EXP_TYPE_10 + self.__equip_exp_1000_105 * EQUIPMENT_EXP_TYPE_1000 + current_exp

        if total_exp < conditions[2]:
            return False
            # raise Exception(
            #     "Lack of exp required to upgrade! total_exp = " + format(total_exp) + " required_exp =" + format(
            #         conditions[2])
            # )

        return True

    # 注册账号、只能领取一次奖励
    # 新手训练500钻石
    # 账号绑定100钻石
    # 关注论坛fb官网150钻石
    # 排行榜奖励1000钻石
    # 总计1750钻石
    def __get_once_reward(self):
        self.__game_request.put_resource('asset_1', 1750)

    # 7日礼包
    def __get_7day_reward(self):
        day = int(self.__quick_reward_count / 2) + 1

        if STAGE_ID_INFO[self.__next_stage] < STAGE_ID_INFO[self.__7day_reward_start_stage]:
            return

        if len(self.__seven_day_rewarded) >= 7:
            return

        for i in range(1, 8):
            if i not in self.__seven_day_rewarded and i <= day:
                self.__game_request.seven_day_gift(day=i)
                self.__seven_day_rewarded.append(i)

    def __get_reward(self):
        reward = ""

        day = int(self.__quick_reward_count / 2) + 1
        if day <= self.__played_days or day > self.__play_day_limit:
            return

        # 每日免费礼包：紫碎*2、钻石*20
        reward = "prop,110,2;diamond,20"

        # 日常活跃奖励
        # 微量钞票2小时、10活跃经验
        reward = reward + ";prop,402,1"
        # 微量力量结晶2小时、15活跃经验
        reward = reward + ";prop,400,1"
        # 蓝碎*5、锦标赛门票*2、20活跃经验
        reward = reward + ";prop,109,5"
        # 钻石100、30活跃经验
        reward = reward + ";diamond,100"
        # 招募令1、50活跃经验
        reward = reward + ";prop,106,1"

        # 竞技场
        self.__game_request.arena_fight(6, self.__arena_count)
        self.__arena_count += 6

        # 派遣
        self.__game_request.get_commission_reward()

        # 公会boss1
        if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__guild_start_stage]:
            combat_dic = {}
            for i in self.__battle_lineup['instance']:
                if i:
                    index = self.__battle_lineup['instance'].index(i)
                    combat_dic[index + 1] = int(i)
            self.__game_request.combat(combat_dic, 3, 1)
            self.__game_request.combat(combat_dic, 3, 1)

        # 活跃通行证
        if day >= self.__master_objective_start_day:
            self.__master_objective_exp += 125
            while self.__master_objective_exp >= MASTER_OBJECTIVE_PERIOD * self.__master_objective_level:
                reward = reward + ";" + MASTER_OBJECTIVE_REWARD[self.__master_objective_level]
                self.__master_objective_level += 1

        # 商店购买
        shop = self.__game_request.get_shop_item_list()
        shop_items = shop.get("userShopItems", [])
        if shop_items:
            reward = reward + ";" + shop_items[0].get("item")

        if day % 2 == 1:
            # 强者之路
            if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__road_to_strong_start_stage]:
                self.__game_request.get_labyrinth_reward()

                # 通行证：经验310、260、55，总计625
                if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__master_road_start_stage]:
                    self.__master_road_exp += 625
                    while self.__master_road_exp >= MASTER_ROAD_PERIOD * self.__master_road_level:
                        reward = reward + ";" + MASTER_ROAD_REWARD[self.__master_road_level]
                        self.__master_road_level += 1

            # 公会boss2
            if STAGE_ID_INFO[self.__next_stage] >= STAGE_ID_INFO[self.__guild_start_stage]:
                combat_dic = {}
                for i in self.__battle_lineup['instance']:
                    if i:
                        index = self.__battle_lineup['instance'].index(i)
                        combat_dic[index + 1] = int(i)
                self.__game_request.combat(combat_dic, 3, 2)
                self.__game_request.combat(combat_dic, 3, 2)

        if day % 7 == 1:
            # 每周免费礼包：5紫碎、20钻石
            reward = reward + ";prop,110,5;diamond,20"

            # 每周活跃奖励：
            # 钞票8小时、复活剂、40活跃经验
            reward = reward + ";prop,461,1"
            # 稀有强化剂100、蓝碎60、复活药剂、80活跃经验
            reward = reward + ";prop,104,100;prop,109,60"
            # 紫碎10、钞票8小时、复活药剂、120活跃经验
            reward = reward + ";prop,110,10;prop,461,1"
            # 钻石400、160活跃经验
            reward = reward + ";diamond,400"
            # 招募令3、复活药剂1、200活跃经验
            reward = reward + ";prop,106,3"

            # 活跃通行证
            if self.__quick_reward_count / 2 >= self.__master_objective_start_day:
                self.__master_objective_exp += 600
                while self.__master_objective_exp >= MASTER_OBJECTIVE_PERIOD * self.__master_objective_level:
                    reward = reward + ";" + MASTER_OBJECTIVE_REWARD[self.__master_objective_level]
                    self.__master_objective_level += 1

        if day % 30 == 1:
            # 每月免费礼包：10紫碎、50钻石
            reward = reward + ";prop,110,10;diamond,50"

        self.__game_request.put_resource_to_user(reward)

        self.__played_days = day

    # 自动穿戴装备
    def __auto_equip(self):
        # 类型: 力量、敏捷、智力
        role_list = [1, 2, 3]

        # 种类: 武器、衣服、裤子、鞋子
        type_list = [1, 2, 3, 4]

        # 各类型装备已穿戴套数
        equiped = {1: 0, 2: 0, 3: 0}

        # 上阵英雄排序
        lineup_hero_info = []
        for hero_instance_id in self.__battle_lineup['instance']:
            if hero_instance_id:
                lineup_hero_info.append(self.__owned_hero_instance[hero_instance_id])
        lineup_hero_info.sort(key=lambda x: (-x['level'], -x['quality']))

        # 按顺序穿戴装备
        for hero_instance in lineup_hero_info:
            equip_ids = []
            role = HERO_LIST[hero_instance['heroId']][2]
            for type in type_list:
                if type not in self.__owned_equip_template:
                    continue
                elif role not in self.__owned_equip_template[type]:
                    continue
                elif len(self.__owned_equip_template[type][role]) <= equiped[role]:
                    continue
                else:
                    equip_ids.append(self.__owned_equip_template[type][role][equiped[role]]['id'])

            if equip_ids:
                equiped[role] += 1
                self.__game_request.hero_equip_on(hero_instance['id'], equip_ids)


if __name__ == '__main__':
    env = EnvironmentMpsen()
    env.reset()
    for i in range(0, 2000):
        legal_action = env.get_legal_action_space()
        action_index = random.randint(0, len(legal_action) - 1)
        ob, reward, done, info = env.step(legal_action[action_index])
        print("action:{}, next stage:{}, reward:{}, done:{}".format(legal_action[action_index], ob[27], reward, done))
