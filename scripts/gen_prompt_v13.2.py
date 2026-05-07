import json
import os
import random
import re
from collections import Counter, defaultdict

# ── 1. 实体库 ───────────────────────────────���───────────────────���─────────────

HUMANS = [
    "adult_man_beard", "adult_woman_curly_hair", "elderly_black_man", "elderly_black_woman",
    "elderly_east_asian_man", "elderly_east_asian_woman", "elderly_white_man", "elderly_white_woman",
    "man_black_suit", "man_bomber_jacket", "man_denim_jacket", "man_flannel_shirt", "man_sportswear",
    "middle_eastern_man", "middle_eastern_woman_hijab", "middleaged_black_man", "middleaged_black_woman_short_hair",
    "south_asian_man", "south_asian_woman", "teen_boy", "teen_girl", "woman_green_cardigan",
    "woman_hijab_coat", "woman_red_hoodie", "woman_white_blazer", "woman_yellow_dress",
    "young_east_asian_man", "young_east_asian_woman", "young_white_man", "young_white_woman",
]

ANIMALS = [
    "black_cat", "brown_horse", "deer", "duck", "elephant", "giraffe", "goat",
    "golden_retriever", "koala", "lion", "owl", "panda", "parrot", "penguin",
    "rabbit", "red_fox", "sheep", "tiger", "wolf", "zebra",
]

FOODS = [
    "apple", "bread_loaf", "burger", "donut", "fried_chicken",
    "green_salad", "ice_cream", "pizza_slice", "spaghetti", "sushi_set",
]

OBJECTS = [
    "acoustic_guitar", "blue_bicycle", "dslr_camera", "folding_umbrella", "headphones",
    "helmet", "open_silver_laptop", "orange_basketball", "red_backpack", "skateboard",
    "soccer_ball", "succulent_plant", "table_lamp", "tennis_racket", "toolbox",
    "travel_suitcase", "tripod", "watering_can", "white_mug", "wooden_dining_chair",
]

NON_HUMANS = ANIMALS + FOODS + OBJECTS

ANIMAL_SET = set(ANIMALS)
FOOD_SET   = set(FOODS)
OBJECT_SET = set(OBJECTS)

BIRD_ANIMALS = {"duck", "owl", "parrot", "penguin"}
PAW_ANIMALS = {"black_cat", "koala", "lion", "panda", "rabbit", "red_fox", "tiger", "wolf"}
NUZZLE_ANIMALS = {"brown_horse", "deer", "elephant", "goat", "golden_retriever", "sheep", "zebra"}

def entity_type(name):
    if name in ANIMAL_SET: return "animal"
    if name in FOOD_SET:   return "food"
    return "object"

# ── 2. 中文名映射 ─────────────────────────────────────────────────────────────

HUMAN_ZH = {
    "adult_man_beard": "留胡子的成年男性",
    "adult_woman_curly_hair": "卷发成年女性",
    "elderly_black_man": "黑人老年男性",
    "elderly_black_woman": "黑人老年女性",
    "elderly_east_asian_man": "东亚老年男性",
    "elderly_east_asian_woman": "东亚老年女性",
    "elderly_white_man": "白人老年男性",
    "elderly_white_woman": "白人老年女性",
    "man_black_suit": "穿黑色西装的男性",
    "man_bomber_jacket": "穿飞行员夹克的男性",
    "man_denim_jacket": "穿牛仔外套的男性",
    "man_flannel_shirt": "穿法兰绒格子衬衫的男性",
    "man_sportswear": "穿运动服的男性",
    "middle_eastern_man": "中东男性",
    "middle_eastern_woman_hijab": "戴头巾的中东女性",
    "middleaged_black_man": "黑人中年男性",
    "middleaged_black_woman_short_hair": "黑人短发中年女性",
    "south_asian_man": "南亚男性",
    "south_asian_woman": "南亚女性",
    "teen_boy": "少年",
    "teen_girl": "少女",
    "woman_green_cardigan": "穿绿色开衫的女性",
    "woman_hijab_coat": "穿外套戴头巾的女性",
    "woman_red_hoodie": "穿红色连帽衫的女性",
    "woman_white_blazer": "穿白色西装外套的女性",
    "woman_yellow_dress": "穿黄色连衣裙的女性",
    "young_east_asian_man": "东亚青年男性",
    "young_east_asian_woman": "东亚青年女性",
    "young_white_man": "白人青年男性",
    "young_white_woman": "白人青年女性",
}

NON_HUMAN_ZH = {
    # animals
    "black_cat": "黑猫", "brown_horse": "棕色马", "deer": "鹿", "duck": "鸭子",
    "elephant": "大象", "giraffe": "长颈鹿", "goat": "山羊", "golden_retriever": "金毛犬",
    "koala": "考拉", "lion": "狮子", "owl": "猫头鹰", "panda": "熊猫", "parrot": "鹦鹉",
    "penguin": "企鹅", "rabbit": "兔子", "red_fox": "红狐", "sheep": "绵羊",
    "tiger": "老虎", "wolf": "狼", "zebra": "斑马",
    # foods
    "apple": "苹果", "bread_loaf": "面包", "burger": "汉堡", "donut": "甜甜圈",
    "fried_chicken": "炸鸡", "green_salad": "绿色沙拉", "ice_cream": "冰淇淋",
    "pizza_slice": "披萨", "spaghetti": "意面", "sushi_set": "寿司拼盘",
    # objects
    "acoustic_guitar": "原声吉他", "blue_bicycle": "蓝色自行车", "dslr_camera": "单反相机",
    "folding_umbrella": "折叠雨伞", "headphones": "耳机", "helmet": "头盔",
    "open_silver_laptop": "打开的银色笔记本电脑", "orange_basketball": "橙色篮球",
    "red_backpack": "红色背包", "skateboard": "滑板", "soccer_ball": "足球",
    "succulent_plant": "多肉植物", "table_lamp": "台灯", "tennis_racket": "网球拍",
    "toolbox": "工具箱", "travel_suitcase": "旅行箱", "tripod": "三脚架",
    "watering_can": "浇水壶", "white_mug": "白色马克杯", "wooden_dining_chair": "木质餐椅",
}

# ── 3. 动作模板 ───────────────────────────────────────────────────────────────
# 每条: (en_template, zh_template)  {h}=human  {o}=object/animal/food

H_FOOD = {
    "burger":       [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} unwraps the {o}.", "{h}打开{o}的包装。"),
                     ("{h} examines the {o}.", "{h}端详着{o}。"),
                     ("{h} sniffs the {o}.", "{h}闻着{o}。")],
    "pizza_slice":  [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} raises the {o}.", "{h}举起{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} examines the {o}.", "{h}端详着{o}。"),
                     ("{h} sniffs the {o}.", "{h}闻着{o}。")],
    "sushi_set":    [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} presents the {o}.", "{h}展示着{o}。"),
                     ("{h} inspects the {o}.", "{h}仔细查看{o}。"),
                     ("{h} picks up the {o}.", "{h}拿起{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} sniffs the {o}.", "{h}闻着{o}。")],
    "green_salad":  [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} serves the {o}.", "{h}端着{o}。"),
                     ("{h} tosses the {o}.", "{h}翻拌着{o}。"),
                     ("{h} examines the {o}.", "{h}端详着{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。")],
    "spaghetti":    [("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} serves the {o}.", "{h}端着{o}。"),
                     ("{h} twirls the {o}.", "{h}卷着{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} sniffs the {o}.", "{h}闻着{o}。")],
    "fried_chicken":[("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} bites the {o}.", "{h}咬着{o}。"),
                     ("{h} raises the {o}.", "{h}举起{o}。"),
                     ("{h} sniffs the {o}.", "{h}闻着{o}。")],
    "donut":        [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} raises the {o}.", "{h}举起{o}。"),
                     ("{h} examines the {o}.", "{h}端详着{o}。"),
                     ("{h} sniffs the {o}.", "{h}闻着{o}。")],
    "ice_cream":    [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} raises the {o}.", "{h}举起{o}。"),
                     ("{h} licks the {o}.", "{h}舔着{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} examines the {o}.", "{h}端详着{o}。")],
    "apple":        [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} eats the {o}.", "{h}吃着{o}。"),
                     ("{h} inspects the {o}.", "{h}仔细查看{o}。"),
                     ("{h} passes the {o}.", "{h}递出{o}。"),
                     ("{h} bites into the {o}.", "{h}咬了一口{o}。"),
                     ("{h} polishes the {o}.", "{h}擦拭着{o}。"),
                     ("{h} raises the {o}.", "{h}举起{o}。")],
    "bread_loaf":   [("{h} holds the {o}.", "{h}拿着{o}。"),
                     ("{h} tears the {o}.", "{h}撕开{o}。"),
                     ("{h} offers the {o}.", "{h}递出{o}。"),
                     ("{h} slices the {o}.", "{h}切着{o}。"),
                     ("{h} smells the {o}.", "{h}闻着{o}。"),
                     ("{h} carries the {o}.", "{h}拿着{o}。")],
}

H_ANIMAL = {
    "golden_retriever": [("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} hugs the {o}.", "{h}抱着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} holds the leash of the {o}.", "{h}牵着{o}的绳子。"),
                          ("{h} plays with the {o}.", "{h}和{o}玩耍。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} scratches the {o}.", "{h}挠着{o}。")],
    "black_cat":        [("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} holds the {o}.", "{h}抱着{o}。"),
                          ("{h} strokes the {o}.", "{h}轻抚{o}。"),
                          ("{h} cradles the {o}.", "{h}轻轻托着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} scratches the {o}.", "{h}挠着{o}。")],
    "brown_horse":      [("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} brushes the {o}.", "{h}给{o}梳毛。"),
                          ("{h} leads the {o}.", "{h}牵着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} touches the {o}.", "{h}触碰着{o}。"),
                          ("{h} walks beside the {o}.", "{h}走在{o}旁边。")],
    "rabbit":           [("{h} holds the {o}.", "{h}抱着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}伸手触碰{o}。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} cradles the {o}.", "{h}轻轻托着{o}。"),
                          ("{h} strokes the {o}.", "{h}轻抚{o}。")],
    "red_fox":          [("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}伸手触碰{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。")],
    "panda":            [("{h} holds the {o}.", "{h}抱着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。")],
    "koala":            [("{h} holds the {o}.", "{h}抱着{o}。"),
                          ("{h} cradles the {o}.", "{h}轻轻托着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} strokes the {o}.", "{h}轻抚{o}。")],
    "penguin":          [("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} reaches toward the {o}.", "{h}伸手触碰{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。")],
    "tiger":            [("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} guides the {o}.", "{h}引导着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。")],
    "elephant":         [("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} touches the {o}.", "{h}触碰着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。")],
    "giraffe":          [("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} touches the {o}.", "{h}触碰着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。")],
    "zebra":            [("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} touches the {o}.", "{h}触碰着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。")],
    "deer":             [("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。"),
                          ("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。")],
    "sheep":            [("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} strokes the {o}.", "{h}轻抚{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} touches the {o}.", "{h}触碰着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。")],
    "goat":             [("{h} pets the {o}.", "{h}抚摸着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} strokes the {o}.", "{h}轻抚{o}。"),
                          ("{h} touches the {o}.", "{h}触碰着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。")],
    "duck":             [("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。")],
    "parrot":           [("{h} holds the {o}.", "{h}托着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。")],
    "owl":              [("{h} holds the {o}.", "{h}托着{o}。"),
                          ("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} feeds the {o}.", "{h}喂着{o}。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。")],
    "lion":             [("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} guides the {o}.", "{h}引导着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。")],
    "wolf":             [("{h} looks at the {o}.", "{h}注视着{o}。"),
                          ("{h} guides the {o}.", "{h}引导着{o}。"),
                          ("{h} watches the {o}.", "{h}凝视着{o}。"),
                          ("{h} crouches beside the {o}.", "{h}蹲在{o}旁边。"),
                          ("{h} gestures toward the {o}.", "{h}朝{o}做手势。"),
                          ("{h} reaches toward the {o}.", "{h}向{o}伸手。")],
}

H_OBJECT = {
    "red_backpack":       [("{h} wears the {o}.", "{h}背着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} lifts the {o}.", "{h}提起{o}。"),
                            ("{h} slings the {o}.", "{h}挎上{o}。"),
                            ("{h} sets down the {o}.", "{h}放下{o}。")],
    "open_silver_laptop": [("{h} types on the {o}.", "{h}在{o}上打字。"),
                            ("{h} looks at the {o}.", "{h}看着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} closes the {o}.", "{h}合上{o}。"),
                            ("{h} points at the {o}.", "{h}指着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。")],
    "orange_basketball":  [("{h} dribbles the {o}.", "{h}运着{o}。"),
                            ("{h} carries the {o}.", "{h}抱着{o}。"),
                            ("{h} spins the {o}.", "{h}转着{o}。"),
                            ("{h} bounces the {o}.", "{h}拍着{o}。"),
                            ("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} raises the {o}.", "{h}举起{o}。")],
    "wooden_dining_chair":[("{h} sits on the {o}.", "{h}坐在{o}上。"),
                            ("{h} rests a hand on the {o}.", "{h}把手放在{o}上。"),
                            ("{h} pulls the {o}.", "{h}拉着{o}。"),
                            ("{h} lifts the {o}.", "{h}搬起{o}。"),
                            ("{h} turns the {o}.", "{h}转动着{o}。"),
                            ("{h} leans on the {o}.", "{h}靠着{o}。")],
    "white_mug":          [("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} drinks from the {o}.", "{h}从{o}里喝东西。"),
                            ("{h} raises the {o}.", "{h}举起{o}。"),
                            ("{h} sips from the {o}.", "{h}小口喝着{o}。"),
                            ("{h} sets down the {o}.", "{h}放下{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。")],
    "blue_bicycle":       [("{h} rides the {o}.", "{h}骑着{o}。"),
                            ("{h} pushes the {o}.", "{h}推着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} leans on the {o}.", "{h}靠着{o}。"),
                            ("{h} walks beside the {o}.", "{h}走在{o}旁边。"),
                            ("{h} locks the {o}.", "{h}锁上{o}。")],
    "dslr_camera":        [("{h} uses the {o}.", "{h}使用着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} raises the {o}.", "{h}举起{o}。"),
                            ("{h} photographs with the {o}.", "{h}用{o}拍照。"),
                            ("{h} holds up the {o}.", "{h}举着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。")],
    "succulent_plant":    [("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} waters the {o}.", "{h}给{o}浇水。"),
                            ("{h} looks at the {o}.", "{h}注视着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。"),
                            ("{h} lifts the {o}.", "{h}举起{o}。"),
                            ("{h} moves the {o}.", "{h}移动着{o}。")],
    "skateboard":         [("{h} rides the {o}.", "{h}踩着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} rests a foot on the {o}.", "{h}把脚放在{o}上。"),
                            ("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。"),
                            ("{h} spins the {o}.", "{h}转动着{o}。")],
    "acoustic_guitar":    [("{h} strums the {o}.", "{h}弹奏着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} tunes the {o}.", "{h}调音着{o}。"),
                            ("{h} raises the {o}.", "{h}举起{o}。")],
    "tennis_racket":      [("{h} swings the {o}.", "{h}挥舞着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} raises the {o}.", "{h}举起{o}。"),
                            ("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} twirls the {o}.", "{h}转动着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。")],
    "table_lamp":         [("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} looks at the {o}.", "{h}注视着{o}。"),
                            ("{h} straightens the {o}.", "{h}扶正{o}。"),
                            ("{h} lifts the {o}.", "{h}提起{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。")],
    "travel_suitcase":    [("{h} pulls the {o}.", "{h}拉着{o}。"),
                            ("{h} opens the {o}.", "{h}打开{o}。"),
                            ("{h} rests a hand on the {o}.", "{h}把手放在{o}上。"),
                            ("{h} lifts the {o}.", "{h}提起{o}。"),
                            ("{h} inspects the {o}.", "{h}检查着{o}。"),
                            ("{h} closes the {o}.", "{h}关上{o}。")],
    "helmet":             [("{h} wears the {o}.", "{h}戴着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} holds up the {o}.", "{h}举着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。"),
                            ("{h} puts on the {o}.", "{h}戴上{o}。")],
    "watering_can":       [("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} waters with the {o}.", "{h}用{o}浇水。"),
                            ("{h} lifts the {o}.", "{h}提起{o}。"),
                            ("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} sets down the {o}.", "{h}放下{o}。"),
                            ("{h} tilts the {o}.", "{h}倾斜着{o}。")],
    "tripod":             [("{h} sets up the {o}.", "{h}架起{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} steadies the {o}.", "{h}稳住{o}。"),
                            ("{h} folds the {o}.", "{h}折叠着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。")],
    "soccer_ball":        [("{h} kicks the {o}.", "{h}踢着{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} rolls the {o}.", "{h}滚着{o}。"),
                            ("{h} holds the {o}.", "{h}抱着{o}。"),
                            ("{h} bounces the {o}.", "{h}拍着{o}。"),
                            ("{h} dribbles the {o}.", "{h}运着{o}。")],
    "toolbox":            [("{h} opens the {o}.", "{h}打开{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} lifts the {o}.", "{h}提起{o}。"),
                            ("{h} rummages through the {o}.", "{h}翻找着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。"),
                            ("{h} sets down the {o}.", "{h}放下{o}。")],
    "folding_umbrella":   [("{h} opens the {o}.", "{h}撑开{o}。"),
                            ("{h} carries the {o}.", "{h}拿着{o}。"),
                            ("{h} folds the {o}.", "{h}折叠{o}。"),
                            ("{h} holds up the {o}.", "{h}举着{o}。"),
                            ("{h} twirls the {o}.", "{h}转动着{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。")],
    "headphones":         [("{h} wears the {o}.", "{h}戴着{o}。"),
                            ("{h} holds the {o}.", "{h}拿着{o}。"),
                            ("{h} adjusts the {o}.", "{h}调整着{o}。"),
                            ("{h} puts on the {o}.", "{h}戴上{o}。"),
                            ("{h} removes the {o}.", "{h}摘下{o}。"),
                            ("{h} examines the {o}.", "{h}端详着{o}。")],
}

# Animal → Food/Object
A_FOOD = {
    "golden_retriever": [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "black_cat":        [("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "rabbit":           [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "duck":             [("{o} pecks at the {f}.", "{o}啄着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。")],
    "parrot":           [("{o} pecks at the {f}.", "{o}啄着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。")],
    "goat":             [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} paws at the {f}.", "{o}用蹄子拨弄{f}。")],
    "sheep":            [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "deer":             [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "panda":            [("{o} holds the {f}.", "{o}拿着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "elephant":         [("{o} lifts the {f}.", "{o}举起{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。")],
    "giraffe":          [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} chews the {f}.", "{o}咀嚼着{f}。")],
    "zebra":            [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "brown_horse":      [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} chews the {f}.", "{o}咀嚼着{f}。")],
    "koala":            [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} licks the {f}.", "{o}舔着{f}。")],
    "lion":             [("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。")],
    "owl":              [("{o} pecks at the {f}.", "{o}啄着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。")],
    "penguin":          [("{o} pecks at the {f}.", "{o}啄着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。")],
    "red_fox":          [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。")],
    "tiger":            [("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。")],
    "wolf":             [("{o} nibbles the {f}.", "{o}啃着{f}。"),
                          ("{o} sniffs the {f}.", "{o}嗅着{f}。"),
                          ("{o} paws at the {f}.", "{o}用爪子拨弄{f}。")],
}

A_OBJECT = {
    "black_cat":        [("{o} sits on the {obj}.", "{o}坐在{obj}上。"),
                          ("{o} curls up on the {obj}.", "{o}蜷缩在{obj}上。"),
                          ("{o} rests beside the {obj}.", "{o}卧在{obj}旁边。"),
                          ("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。")],
    "golden_retriever": [("{o} rests beside the {obj}.", "{o}卧在{obj}旁边。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。")],
    "rabbit":           [("{o} sits beside the {obj}.", "{o}坐在{obj}旁边。"),
                          ("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。")],
    "parrot":           [("{o} perches on the {obj}.", "{o}栖息在{obj}上。"),
                          ("{o} pecks at the {obj}.", "{o}啄着{obj}。"),
                          ("{o} stands beside the {obj}.", "{o}站在{obj}旁边。")],
    "owl":              [("{o} perches on the {obj}.", "{o}栖息在{obj}上。"),
                          ("{o} rests beside the {obj}.", "{o}卧在{obj}旁边。"),
                          ("{o} stands beside the {obj}.", "{o}站在{obj}旁边。")],
    "duck":             [("{o} stands beside the {obj}.", "{o}站在{obj}旁边。"),
                          ("{o} waddles around the {obj}.", "{o}围着{obj}走动。"),
                          ("{o} pecks at the {obj}.", "{o}啄着{obj}。")],
    "goat":             [("{o} nudges the {obj}.", "{o}用角顶着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用蹄子刨着{obj}。"),
                          ("{o} stands beside the {obj}.", "{o}站在{obj}旁边。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。")],
    "sheep":            [("{o} rests beside the {obj}.", "{o}卧在{obj}旁边。"),
                          ("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。")],
    "panda":            [("{o} rests against the {obj}.", "{o}靠着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。"),
                          ("{o} sits beside the {obj}.", "{o}坐在{obj}旁边。")],
    "koala":            [("{o} rests against the {obj}.", "{o}靠着{obj}。"),
                          ("{o} grips the {obj}.", "{o}抓住{obj}。"),
                          ("{o} sits beside the {obj}.", "{o}坐在{obj}旁边。")],
    "brown_horse":      [("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。")],
    "deer":             [("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用蹄子拨弄{obj}。")],
    "elephant":         [("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} carries the {obj}.", "{o}卷起{obj}。")],
    "giraffe":          [("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用蹄子拨弄{obj}。")],
    "lion":             [("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。"),
                          ("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。")],
    "penguin":          [("{o} pecks at the {obj}.", "{o}啄着{obj}。"),
                          ("{o} nudges the {obj}.", "{o}用喙顶着{obj}。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。")],
    "red_fox":          [("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。"),
                          ("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。")],
    "tiger":            [("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。"),
                          ("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。")],
    "wolf":             [("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。"),
                          ("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} carries the {obj}.", "{o}叼着{obj}。")],
    "zebra":            [("{o} nudges the {obj}.", "{o}用鼻子蹭着{obj}。"),
                          ("{o} sniffs the {obj}.", "{o}嗅着{obj}。"),
                          ("{o} paws at the {obj}.", "{o}用蹄子拨弄{obj}。")],
}

# Object-object relationships (承托/附着)
OBJ_OBJ = [
    ("headphones", "wooden_dining_chair",
     "The headphones hang on the wooden dining chair.",
     "耳机挂在木质餐椅上。"),
    ("folding_umbrella", "travel_suitcase",
     "The folding umbrella leans against the travel suitcase.",
     "折叠雨伞靠在旅行箱上。"),
    ("white_mug", "toolbox",
     "The white mug is on the toolbox.",
     "白色马克杯放在工具箱上。"),
    ("helmet", "travel_suitcase",
     "The helmet rests on the travel suitcase.",
     "头盔放在旅行箱上。"),
    ("tripod", "toolbox",
     "The tripod is beside the toolbox.",
     "三脚架在工具箱旁边。"),
    ("white_mug", "wooden_dining_chair",
     "The white mug is on the wooden dining chair.",
     "白色马克杯放在木质餐椅上。"),
]

OO_STATIC_TEMPLATES = [
    ("The {subj} is to the left of the {occ}.", "{subj}在{occ}左边。", "left_of"),
    ("The {subj} is to the right of the {occ}.", "{subj}在{occ}右边。", "right_of"),
    ("The {subj} is beside the {occ}.", "{subj}在{occ}旁边。", "beside"),
]

# Human-Human (only occlusion_interaction)
HH_TEMPLATES = [
    ("{h1} and {h2} carry the {obj} together.", "{h1}和{h2}一起搬着{obj}。"),
    ("{h1} and {h2} open the {obj}.", "{h1}和{h2}一起打开{obj}。"),
    ("{h1} high-fives {h2}.", "{h1}与{h2}击掌。"),
    ("{h1} fist-bumps {h2}.", "{h1}与{h2}碰拳。"),
    ("{h1} hugs {h2}.", "{h1}拥抱{h2}。"),
    ("{h1} pats {h2}.", "{h1}轻拍{h2}。"),
    ("{h1} holds hands with {h2}.", "{h1}与{h2}牵手。"),
]

# 遮挡体（有合理体积，可遮挡人/动物）
VALID_OCCLUDERS = {
    # Large objects that can physically occlude a person
    "blue_bicycle", "travel_suitcase", "wooden_dining_chair",
    # Large animals
    "brown_horse", "elephant", "zebra", "giraffe",
}

OCCLUSION_TEMPLATES = [
    ("{subj} is partly hidden by the {occ}.", "{subj}被{occ}部分遮挡。"),
    ("{subj} stands behind the {occ}, partly hidden.", "{subj}站在{occ}后面，部分身体被遮挡。"),
]

HO_OCC_LARGE_PARTIAL = [
    ("The {occ} blocks part of {subj}'s upper body.", "{occ}挡住了{subj}上半身的一部分。"),
    ("The {occ} covers part of {subj}'s lower body.", "{occ}遮住了{subj}下半身的一部分。"),
    ("The {occ} hides part of {subj}'s torso.", "{occ}挡住了{subj}躯干的一部分。"),
    ("The {occ} blocks {subj}'s legs.", "{occ}挡住了{subj}的腿部。"),
    ("The {occ} obscures part of {subj}'s arm area.", "{occ}遮住了{subj}手臂区域的一部分。"),
    ("{subj} stands behind the {occ}, with the lower body partly blocked.", "{subj}站在{occ}后面，下半身部分被挡住。"),
    ("{subj} stands behind the {occ}, with the upper body partly obscured.", "{subj}站在{occ}后面，上半身部分被遮住。"),
]

HO_OCC_SMALL_GROUND = [
    ("The {occ} blocks part of {subj}'s feet.", "{occ}挡住了{subj}脚部的一部分。"),
    ("The {occ} obscures {subj}'s lower legs.", "{occ}遮住了{subj}小腿的一部分。"),
    ("The {occ} blocks part of {subj}'s lower body.", "{occ}挡住了{subj}下半身的一部分。"),
    ("{subj} is visible behind the {occ}, with the feet partly blocked.", "{subj}在{occ}后方可见，但脚部部分被挡住。"),
]

# ── 4. ratio 映射 ──────────────────────────────��──────────────────────────��───
RATIO_MAP = {
    "human_heavy":  {8: (6, 2), 6: (4, 2), 4: (3, 1), 2: (2, 0)},
    "balanced":     {8: (4, 4), 6: (3, 3), 4: (2, 2), 2: (1, 1)},
    "object_heavy": {8: (2, 6), 6: (2, 4), 4: (1, 3), 2: (0, 2)},
}

# ── 5. 辅助函数 ───────────────────────────────────────────────────────────────
def hn(name):    return name.replace("_", " ")
def on_(name):   return name.replace("_", " ")
def hn_zh(name): return HUMAN_ZH[name]
def on_zh(name): return NON_HUMAN_ZH[name]


def occ_desc(name, allow_small_ground=True):
    en = on_(name)
    zh = on_zh(name)
    if name in VALID_OCCLUDERS or not allow_small_ground:
        return en, zh
    return f"{en} on the ground", f"地面上的{zh}"

def pick(lst, idx): return lst[idx % len(lst)]

def est_tokens(text): return int(len(text.split()) * 1.1)

WEAK_OCCLUSION_INTERACTION_SUBSTRINGS = (
    "reaches toward",
    "looks at",
    "glances at",
    "faces",
    "approaches",
    "steps toward",
    "leans toward",
)


def _filter_occlusion_interaction_templates(templates):
    filtered = [
        t for t in templates
        if not any(s in t[0] for s in WEAK_OCCLUSION_INTERACTION_SUBSTRINGS)
    ]
    return filtered if filtered else templates

# 语法复数名词（主语时用 are 而非 is）
PLURAL_ENTITIES = {"headphones"}

def verb_be(name):
    return "are" if name in PLURAL_ENTITIES else "is"

def pick_ref(candidates, exclude, held=None, strict=False):
    """Pick best spatial reference: prefer ground-level items; avoid held objects.
    strict=True: return None if no grounded ref (used in overflow loop — let final fallback handle it).
    strict=False: fall back to held items as last resort (used in final fallback).
    Food items that are NOT held are treated as valid ground references (they can sit on the floor/table)."""
    held = held or set()
    # Tier 1: grounded non-food (not held, not food)
    grounded = [r for r in candidates if r != exclude and r not in held and entity_type(r) != "food"]
    if grounded:
        return grounded[0]
    # Tier 1b: non-held food (not carried — valid as ground/table prop)
    non_held_food = [r for r in candidates if r != exclude and r not in held and entity_type(r) == "food"]
    if non_held_food:
        return non_held_food[0]
    if strict:
        return None
    # Tier 2: not food but may be held (last resort — avoid but accept if no alternative)
    not_food = [r for r in candidates if r != exclude and entity_type(r) != "food"]
    if not_food:
        return not_food[0]
    # Tier 3: anything not excluded
    return next((r for r in candidates if r != exclude), None)

# no_interaction 两个 tag 专用：纯位置/静态，无物理接触
# 适用于所有 non-human 实体类型（food / animal / object 统一用位置词）
STATIC_H_ENTITY = [
    ("{h} stands near the {o}.", "{h}站在{o}附近。"),
    ("{h} stands beside the {o}.", "{h}站在{o}旁边。"),
    ("{h} looks at the {o}.", "{h}注视着{o}。"),
    ("{h} faces the {o}.", "{h}面向{o}。"),
    ("{h} gazes at the {o}.", "{h}凝视着{o}。"),
    ("{h} observes the {o}.", "{h}观察着{o}。"),
    ("{h} glances at the {o}.", "{h}瞥向{o}。"),
    ("{h} approaches the {o}.", "{h}走向{o}。"),
    ("{h} steps toward the {o}.", "{h}朝{o}走近。"),
    ("{h} leans toward the {o}.", "{h}向{o}倾身。"),
    ("{h} pauses beside the {o}.", "{h}在{o}旁边停下。"),
]

# Human-Human templates for v12 relation control
HH_STATIC_TEMPLATES = [
    ("{h1} stands beside {h2}.", "{h1}站在{h2}旁边。"),
    ("{h1} looks at {h2}.", "{h1}注视着{h2}。"),
    ("{h1} faces {h2}.", "{h1}面向{h2}。"),
    ("{h1} glances at {h2}.", "{h1}瞥向{h2}。"),
    ("{h1} walks toward {h2}.", "{h1}朝{h2}走去。"),
    ("{h1} gazes at {h2}.", "{h1}凝视着{h2}。"),
    ("{h1} turns toward {h2}.", "{h1}转向{h2}。"),
]

HH_OCC_TEMPLATES = [
    ("{h1} stands behind {h2}, partly hidden.", "{h1}站在{h2}身后，被部分遮挡。"),
    ("{h1} is partly hidden behind {h2}.", "{h1}躲在{h2}后面，被部分遮挡。"),
    ("{h1} is behind {h2}, partially obscured by {h2}.", "{h1}在{h2}后面，被{h2}部分遮住。"),
    ("{h1} stands in front of {h2}, partly blocking {h2}.", "{h1}站在{h2}前面，部分遮住{h2}。"),
]

HH_INTERACTION_TEMPLATES = [
    ("{h1} high-fives {h2}.", "{h1}与{h2}击掌。"),
    ("{h1} fist-bumps {h2}.", "{h1}与{h2}碰拳。"),
    ("{h1} hugs {h2}.", "{h1}拥抱{h2}。"),
    ("{h1} holds hands with {h2}.", "{h1}与{h2}牵手。"),
    ("{h1} shakes hands with {h2}.", "{h1}与{h2}握手。"),
    ("{h1} taps {h2} on the shoulder.", "{h1}轻拍{h2}的肩膀。"),
    ("{h1} pats {h2} on the back.", "{h1}轻拍{h2}的后背。"),
    ("{h1} links arms with {h2}.", "{h1}与{h2}挽着手臂。"),
    ("{h1} walks arm in arm with {h2}.", "{h1}与{h2}挽臂同行。"),
    ("{h1} places a hand on {h2}'s shoulder.", "{h1}把手放在{h2}肩上。"),
    ("{h1} pulls {h2} closer.", "{h1}把{h2}拉近。"),
    ("{h1} leans on {h2}.", "{h1}倚靠着{h2}。"),
]

REL_CATS = ("HHi", "HHs", "HHocc", "HOi", "HOs", "HOocc", "OOs", "OOocc", "AOi", "NOi", "OOi")

# v13 slot plan: allocate entity-disjoint core relations first, then realize them as sentences.
SLOT_PLAN = {
    ("balanced", "no_interaction_no_occlusion"): {
        8: {"HOs": 4},
        6: {"HOs": 3},
        4: {"HOs": 2},
        2: {"HOs": 1},
    },
    ("balanced", "occlusion_no_interaction"): {
        8: {"HOocc": 4},
        6: {"HOocc": 3},
        4: {"HOocc": 2},
        2: {"HOocc": 1},
    },
    ("balanced", "occlusion_interaction"): {
        8: {"HOi": 4},
        6: {"HOi": 3},
        4: {"HOi": 2},
        2: {"HOi": 1},
    },
    ("human_heavy", "no_interaction_no_occlusion"): {
        8: {"HOs": 2, "HHs": 2},
        6: {"HOs": 2, "HHs": 1},
        4: {"HOs": 1, "HHs": 1},
        2: {"HHs": 1},
    },
    ("human_heavy", "occlusion_no_interaction"): {
        8: {"HOocc": 2, "HHocc": 2},
        6: {"HOocc": 2, "HHocc": 1},
        4: {"HOocc": 1, "HHocc": 1},
        2: {"HHocc": 1},
    },
    ("human_heavy", "occlusion_interaction"): {
        8: {"HOi": 2, "HHi": 2},
        6: {"HOi": 2, "HHi": 1},
        4: {"HOi": 1, "HHi": 1},
        2: {"HHi": 1},
    },
    ("object_heavy", "no_interaction_no_occlusion"): {
        8: {"HOs": 2, "OOs": 2},
        6: {"HOs": 2, "OOs": 1},
        4: {"HOs": 1, "OOs": 1},
        2: {"OOs": 1},
    },
    ("object_heavy", "occlusion_no_interaction"): {
        8: {"HOocc": 2, "OOocc": 2},
        6: {"HOocc": 2, "OOocc": 1},
        4: {"HOocc": 1, "OOocc": 1},
        2: {"OOocc": 1},
    },
    ("object_heavy", "occlusion_interaction"): {
        8: {"HOi": 2, "NOi": 2},
        6: {"HOi": 2, "NOi": 1},
        4: {"HOi": 1, "NOi": 1},
        2: {"NOi": 1},
    },
}

# ── 6. Prompt 生成 ────────────────────────────────────────────────────────────
def build_prompt_logic(tag, seed, selected_h, selected_o):
    """返回句子列表: [(en, zh, frozenset_of_entity_ids), ...]"""
    rng = random.Random(seed * 1000 + len(selected_h) + len(selected_o))

    h_list = list(selected_h)
    o_list = list(selected_o)
    n_h = len(h_list)

    # 按类型分组
    animals  = [o for o in o_list if entity_type(o) == "animal"]
    foods    = [o for o in o_list if entity_type(o) == "food"]
    objects  = [o for o in o_list if entity_type(o) == "object"]

    raw_sents = []   # (en, zh) — 未标注实体引用
    covered_h = set()
    covered_o = set()

    def add(en, zh):
        raw_sents.append((en, zh))

    def fmt_h(name):   return hn(name)
    def fmt_o(name):   return on_(name)
    def fmt_hzh(name): return hn_zh(name)
    def fmt_ozh(name): return on_zh(name)

    use_static = (tag in ("no_interaction_no_occlusion", "occlusion_no_interaction"))

    # 检查 human 个数 = 0 的情况（object_heavy level=2）
    SPATIAL_PAIRS = [
        ("to the left of", "在{r}左边"),
        ("to the right of", "在{r}右边"),
        ("in front of", "在{r}前面"),
        ("behind", "在{r}后面"),
        ("beside", "在{r}旁边"),
    ]

    h_assignments = {}  # populated in else branch; referenced in occlusion section below

    if n_h == 0:
        # 只有 non-human 实体
        # 优先：animal-food, animal-object, OBJ_OBJ；最后才用空间位置关系
        for o in o_list:
            if o in covered_o:
                continue
            otype = entity_type(o)
            if otype == "food":
                eater = next((a for a in animals if a in A_FOOD and a not in covered_o), None)
                if eater and not use_static:
                    t = pick(A_FOOD[eater], rng.randint(0, 99))
                    add(t[0].format(o=fmt_o(eater), f=fmt_o(o)),
                        t[1].format(o=fmt_ozh(eater), f=fmt_ozh(o)))
                    covered_o.add(eater); covered_o.add(o)
                else:
                    shelf = next((ob for ob in objects if ob in o_list and ob not in covered_o), None)
                    if shelf:
                        add(f"The {fmt_o(o)} is on the {fmt_o(shelf)}.",
                            f"{fmt_ozh(o)}放在{fmt_ozh(shelf)}上。")
                        covered_o.add(o); covered_o.add(shelf)
                    # else: leave uncovered, spatial fallback will handle
            elif otype == "animal":
                obj_target = next((ob for ob in objects if ob in o_list and ob not in covered_o
                                   and o in A_OBJECT), None)
                if obj_target and not use_static:
                    t = pick(A_OBJECT[o], rng.randint(0, 99))
                    add(t[0].format(o=fmt_o(o), obj=fmt_o(obj_target)),
                        t[1].format(o=fmt_ozh(o), obj=fmt_ozh(obj_target)))
                    covered_o.add(o); covered_o.add(obj_target)
                # else: leave uncovered, spatial fallback will handle
            else:
                for (a, b, en, zh) in OBJ_OBJ:
                    if (a == o or b == o) and a in o_list and b in o_list:
                        add(en, zh); covered_o.add(a); covered_o.add(b); break

        # 空间位置关系兜底：覆盖还没出现的实体（两两配对）
        uncovered = [o for o in o_list if o not in covered_o]
        already_placed = [o for o in o_list if o in covered_o]
        # 先把 uncovered 里两两配对写相对位置
        i = 0
        while i < len(uncovered):
            if i + 1 < len(uncovered):
                a, b = uncovered[i], uncovered[i + 1]
                sp = SPATIAL_PAIRS[i % len(SPATIAL_PAIRS)]
                add(f"The {fmt_o(a)} {verb_be(a)} {sp[0]} the {fmt_o(b)}.",
                    f"{fmt_ozh(a)}{sp[1].format(r=fmt_ozh(b))}。")
                covered_o.add(a); covered_o.add(b)
                i += 2
            else:
                o = uncovered[i]
                ref = pick_ref(already_placed + [x for x in uncovered if x != o], o)
                if ref and ref != o:
                    sp = SPATIAL_PAIRS[i % len(SPATIAL_PAIRS)]
                    add(f"The {fmt_o(o)} {verb_be(o)} {sp[0]} the {fmt_o(ref)}.",
                        f"{fmt_ozh(o)}{sp[1].format(r=fmt_ozh(ref))}。")
                covered_o.add(o)
                i += 1
    else:
        # ── 正常情况：human 驱动 ─────────────────────────────────────────────
        MAX_PER_HUMAN = 2  # 每人最多做2件事，超出的交给动物自主动作/物体关系
        h_assignments = {h: [] for h in h_list}
        overflow = []  # 超出配额的物体
        # index-aligned: h[i] pairs with o[i], so lower-level subsets retain their sentences
        for i, o in enumerate(o_list[:n_h]):
            h_assignments[h_list[i]].append(o)
        for i, o in enumerate(o_list[n_h:]):
            h = h_list[i % n_h]
            if len(h_assignments[h]) < MAX_PER_HUMAN:
                h_assignments[h].append(o)
            else:
                overflow.append(o)

        # Entities actively held/interacted by humans (non-static tags) — not valid spatial references
        if not use_static:
            held_o = {o for h_objs in h_assignments.values() for o in h_objs}
        else:
            held_o = set()

        FALLBACK_HH_STATIC = [
            ("{h1} stands beside {h2}.", "{h1}站在{h2}旁边。"),
            ("{h1} looks at {h2}.", "{h1}注视着{h2}。"),
            ("{h1} faces {h2}.", "{h1}面向{h2}。"),
            ("{h1} glances at {h2}.", "{h1}瞥向{h2}。"),
            ("{h1} walks toward {h2}.", "{h1}朝{h2}走去。"),
            ("{h1} gazes at {h2}.", "{h1}凝视着{h2}。"),
            ("{h1} turns toward {h2}.", "{h1}转向{h2}。"),
        ]
        # occlusion_no_interaction：部分 unassigned human 用人-人遮挡句（明确表达遮挡关系）
        FALLBACK_HH_OCC = [
            ("{h1} stands behind {h2}, partly hidden.", "{h1}站在{h2}身后，被部分遮挡。"),
            ("{h1} is partly hidden behind {h2}.", "{h1}躲在{h2}后面，被部分遮挡。"),
            ("{h1} is behind {h2}, partially obscured by {h2}.", "{h1}在{h2}后面，被{h2}部分遮住。"),
            ("{h1} stands in front of {h2}, partly blocking {h2}.", "{h1}站在{h2}前面，部分遮住{h2}。"),
        ]
        # ── H-O sentences for assigned humans FIRST — ensures objects appear early in raw_sents,
        # giving them priority in truncation before unassigned-human (H-H) sentences fill the budget.
        for h, assigned in h_assignments.items():
            if not assigned:
                continue

            for i_a, o in enumerate(assigned):
                # For occlusion_no_interaction: skip the static H-O sentence for the anchor pair
                # (h_list[0] + o_list[0]). The occlusion section generates a more specific
                # "hidden by" sentence for this pair — having both "stands near" and "hidden by"
                # for the same person-object pair is redundant/contradictory.
                if (tag == "occlusion_no_interaction" and
                        bool(o_list) and o_list[0] in VALID_OCCLUDERS and
                        h == h_list[0] and o == o_list[0]):
                    covered_o.add(o)
                    covered_h.add(h)
                    continue
                if use_static:
                    # 纯位置/静态：无物理接触
                    t = pick(STATIC_H_ENTITY, rng.randint(0, 99))
                else:
                    # 强交互：用物体专属动作模板
                    otype = entity_type(o)
                    if otype == "food":
                        cands = H_FOOD.get(o, [("{h} holds the {o}.", "{h}拿着{o}。")])
                        cands = _filter_occlusion_interaction_templates(cands)
                        t = pick(cands, rng.randint(0, 99))
                    elif otype == "animal":
                        cands = H_ANIMAL.get(o, [("{h} touches the {o}.", "{h}触碰着{o}。")])
                        cands = _filter_occlusion_interaction_templates(cands)
                        t = pick(cands, rng.randint(0, 99))
                    else:
                        cands = H_OBJECT.get(o, [("{h} carries the {o}.", "{h}拿着{o}。")])
                        cands = _filter_occlusion_interaction_templates(cands)
                        t = pick(cands, rng.randint(0, 99))
                add(t[0].format(h=fmt_h(h), o=fmt_o(o)),
                    t[1].format(h=fmt_hzh(h), o=fmt_ozh(o)))
                covered_h.add(h)
                covered_o.add(o)

        # Anchor object-pair sentence: ensures o[0] and o[1] co-appear at levels where n_h=0
        # Skip for occlusion+object_heavy: the supplement "o[1] behind o[0]" already covers level-2
        # (having both would give contradictory position relationships for the same pair)
        if len(o_list) >= 2 and not (
            tag in ("occlusion_no_interaction", "occlusion_interaction") and n_h < len(o_list)
        ):
            oa, ob = o_list[0], o_list[1]
            ta, tb = entity_type(oa), entity_type(ob)
            handled_oo = False
            if not use_static and ta == "animal" and tb == "food" and oa in A_FOOD:
                t = pick(A_FOOD[oa], rng.randint(0, 99))
                add(t[0].format(o=fmt_o(oa), f=fmt_o(ob)), t[1].format(o=fmt_ozh(oa), f=fmt_ozh(ob)))
                handled_oo = True
            elif not use_static and tb == "animal" and ta == "food" and ob in A_FOOD:
                t = pick(A_FOOD[ob], rng.randint(0, 99))
                add(t[0].format(o=fmt_o(ob), f=fmt_o(oa)), t[1].format(o=fmt_ozh(ob), f=fmt_ozh(oa)))
                handled_oo = True
            if not handled_oo:
                for (a, b, en, zh) in OBJ_OBJ:
                    if {a, b} == {oa, ob}:
                        add(en, zh); handled_oo = True; break
            if not handled_oo:
                en_tpl, zh_tpl, _ = _pick_template(OO_STATIC_TEMPLATES, rng)
                add(en_tpl.format(subj=fmt_o(oa), occ=fmt_o(ob)),
                    zh_tpl.format(subj=fmt_ozh(oa), occ=fmt_ozh(ob)))

        # ── Unassigned humans: H-O (40%) or H-H (60%), all via RNG for verb/partner variety
        unassigned_humans = [h for h, a in h_assignments.items() if not a]
        other_humans_pool = [hh for hh in h_list if hh not in set(unassigned_humans)]
        for h in unassigned_humans:
            # 40% chance: reference o_list[0] (anchor object — present at ALL lower levels).
            # Do NOT reference o_list[1+]: those may not exist at lower levels, causing filtering.
            if o_list and rng.randint(0, 99) < 40:
                t = STATIC_H_ENTITY[rng.randint(0, 99) % len(STATIC_H_ENTITY)]
                add(t[0].format(h=fmt_h(h), o=fmt_o(o_list[0])),
                    t[1].format(h=fmt_hzh(h), o=fmt_ozh(o_list[0])))
            else:
                # Reference another human — pick partner and template via RNG
                preferred = [hh for hh in other_humans_pool if hh != h]
                partner_pool = preferred if preferred else [hh for hh in h_list if hh != h]
                partner = partner_pool[rng.randint(0, 99) % len(partner_pool)]
                t = FALLBACK_HH_STATIC[rng.randint(0, 99) % len(FALLBACK_HH_STATIC)]
                add(t[0].format(h1=fmt_h(h), h2=fmt_h(partner)),
                    t[1].format(h1=fmt_hzh(h), h2=fmt_hzh(partner)))
            covered_h.add(h)

        # Anchor HH: for non-occlusion human_heavy (n_h > n_o), ensures h[0]+h[1] survive level-2.
        # For occlusion tags, a dedicated HH_OCC supplement is generated in the occlusion section
        # below (h[1] hidden by h[0]), which covers both entity presence and occlusion keyword.
        if n_h > len(o_list) and tag not in ("occlusion_no_interaction", "occlusion_interaction"):
            t = FALLBACK_HH_STATIC[rng.randint(0, 99) % len(FALLBACK_HH_STATIC)]
            add(t[0].format(h1=fmt_h(h_list[0]), h2=fmt_h(h_list[1])),
                t[1].format(h1=fmt_hzh(h_list[0]), h2=fmt_hzh(h_list[1])))

        # ── overflow 物体：用动物自主动作或物体-物体关系处理 ─────────────────
        for o in overflow:
            otype = entity_type(o)
            handled = False
            if not use_static and otype == "animal" and o in A_FOOD:
                # 找一个 food 让动物吃
                food_target = next((f for f in foods if f in covered_o), None)
                if food_target:
                    t = pick(A_FOOD[o], rng.randint(0, 99))
                    add(t[0].format(o=fmt_o(o), f=fmt_o(food_target)),
                        t[1].format(o=fmt_ozh(o), f=fmt_ozh(food_target)))
                    covered_o.add(o); handled = True
            if not use_static and not handled and otype == "animal" and o in A_OBJECT:
                obj_target = next((ob for ob in objects if ob in covered_o), None)
                if obj_target:
                    t = pick(A_OBJECT[o], rng.randint(0, 99))
                    add(t[0].format(o=fmt_o(o), obj=fmt_o(obj_target)),
                        t[1].format(o=fmt_ozh(o), obj=fmt_ozh(obj_target)))
                    covered_o.add(o); handled = True
            if not handled:
                # 用静态位置关系兜底：优先选动物/物体作参照，避免食物（食物在人手中，不是地面参照）
                ref = pick_ref(list(covered_o), o, held=held_o, strict=True)
                if ref:
                    sp = SPATIAL_PAIRS[len(covered_o) % len(SPATIAL_PAIRS)]
                    add(f"The {fmt_o(o)} {verb_be(o)} {sp[0]} the {fmt_o(ref)}.",
                        f"{fmt_ozh(o)}{sp[1].format(r=fmt_ozh(ref))}。")
                    covered_o.add(o)
                # else: leave uncovered — final spatial fallback will pair with overflow peers

        # ── object-object 关系补充（覆盖未被 human 直接操作的 object）──────
        still_uncovered = [o for o in o_list if o not in covered_o]
        for o in still_uncovered:
            otype = entity_type(o)
            if otype == "object":
                # 找一个可以承托它的 object（静态放置，无交互，所有 tag 均可用）
                # Skip pairs where the partner object is held — a held object is not a valid static ref
                for (a, b, en, zh) in OBJ_OBJ:
                    if (a == o or b == o) and a in o_list and b in o_list:
                        partner = b if a == o else a
                        if partner in held_o:
                            continue  # partner is carried — find another pair
                        add(en, zh)
                        covered_o.add(a); covered_o.add(b)
                        break
                else:
                    if not use_static:
                        # fallback：找一个已覆盖的 animal 来互动
                        eligible_a = [an for an in animals if an in covered_o and o in A_OBJECT.get(an, {})]
                        if eligible_a:
                            an = eligible_a[0]
                            t = pick(A_OBJECT[an], rng.randint(0, 99))
                            add(t[0].format(o=fmt_o(an), obj=fmt_o(o)),
                                t[1].format(o=fmt_ozh(an), obj=fmt_ozh(o)))
                            covered_o.add(o)
            elif otype == "animal":
                if not use_static:
                    # 动物没被 human 提到：找一个 food 让它吃
                    edible = [f for f in foods if f in covered_o]
                    if edible and o in A_FOOD:
                        f = edible[0]
                        t = pick(A_FOOD[o], rng.randint(0, 99))
                        add(t[0].format(o=fmt_o(o), f=fmt_o(f)),
                            t[1].format(o=fmt_ozh(o), f=fmt_ozh(f)))
                        covered_o.add(o)
                    else:
                        obj_targets = [ob for ob in objects if ob in covered_o and o in A_OBJECT.get(o, {})]
                        if obj_targets and o in A_OBJECT:
                            t = pick(A_OBJECT[o], rng.randint(0, 99))
                            add(t[0].format(o=fmt_o(o), obj=fmt_o(obj_targets[0])),
                                t[1].format(o=fmt_ozh(o), obj=fmt_ozh(obj_targets[0])))
                            covered_o.add(o)
        # Spatial fallback for anything still uncovered (mainly for use_static=True case)
        for i, o in enumerate(o_list):
            if o not in covered_o:
                ref = pick_ref(o_list, o, held=held_o)
                if ref:
                    sp = SPATIAL_PAIRS[i % len(SPATIAL_PAIRS)]
                    add(f"The {fmt_o(o)} {verb_be(o)} {sp[0]} the {fmt_o(ref)}.",
                        f"{fmt_ozh(o)}{sp[1].format(r=fmt_ozh(ref))}。")
                    covered_o.add(ref)  # reference entity is now introduced — prevent duplicate sentence
                covered_o.add(o)

    # ── 遮挡句（anchor 策略：用 o[0]+h[0] 确保低 level 也能保留遮挡句）──────
    if tag in ("occlusion_no_interaction", "occlusion_interaction"):
        valid_anchor = bool(o_list) and o_list[0] in VALID_OCCLUDERS

        # Primary: human hidden by o[0].
        # For interaction tags: prefer a human that doesn't already interact with o[0],
        # so we don't say "h feeds zebra" AND "h is hidden by zebra" (redundant/contradictory).
        # For no_interaction tags: h[0] is the anchor subject (static H-O skipped above).
        if valid_anchor and h_list:
            occ = o_list[0]
            if not use_static and h_assignments:
                # interaction tag: skip humans that are already assigned to interact with o[0]
                non_interacting = [h for h in h_list if occ not in h_assignments.get(h, [])]
                primary_subj = non_interacting[0] if non_interacting else h_list[0]
            else:
                primary_subj = h_list[0]
            t = pick(OCCLUSION_TEMPLATES, rng.randint(0, 1))
            add(t[0].format(subj=fmt_h(primary_subj), occ=fmt_o(occ)),
                t[1].format(subj=fmt_hzh(primary_subj), occ=fmt_ozh(occ)))
            # HH supplement for human_heavy: ensures level-2 (n_o=0) still has an occ sentence.
            # Use the h that is NOT primary_subj as the supplement subject.
            if n_h > len(o_list) and len(h_list) >= 2:
                HH_SUPP_OCC = [
                    ("{subj} is partly hidden by {occ}.", "{subj}被{occ}部分遮挡。"),
                    ("{subj} stands behind {occ}, partly hidden.", "{subj}站在{occ}身后，被部分遮挡。"),
                ]
                supp_subj = h_list[1] if primary_subj == h_list[0] else h_list[0]
                supp_occ_h = h_list[0] if supp_subj == h_list[1] else h_list[1]
                t2 = pick(HH_SUPP_OCC, rng.randint(0, 99) % len(HH_SUPP_OCC))
                add(t2[0].format(subj=fmt_h(supp_subj), occ=fmt_h(supp_occ_h)),
                    t2[1].format(subj=fmt_hzh(supp_subj), occ=fmt_hzh(supp_occ_h)))
        elif not valid_anchor and len(h_list) >= 2:
            # No valid anchor object: h-h fallback
            HH_OCC = [
                ("{subj} is partly hidden by {occ}.", "{subj}被{occ}部分遮挡。"),
                ("{subj} is partly hidden behind {occ}.", "{subj}站在{occ}身后，身体部分被遮挡。"),
            ]
            t = pick(HH_OCC, rng.randint(0, 1))
            add(t[0].format(subj=fmt_h(h_list[0]), occ=fmt_h(h_list[1])),
                t[1].format(subj=fmt_hzh(h_list[0]), occ=fmt_hzh(h_list[1])))

        # object_heavy supplement: o[1] hidden by o[0] — survives level-2 (n_h=0, n_o=2)
        # Use object-appropriate templates (no "stands"/"身体" — objects don't have bodies)
        if n_h < len(o_list) and valid_anchor and len(o_list) >= 2:
            OBJ_OCC_TEMPLATES = [
                ("The {subj} is partly hidden by the {occ}.", "{subj}被{occ}部分遮挡。"),
                ("The {subj} is behind the {occ}.", "{subj}在{occ}后面。"),
                ("The {subj} is partly obscured by the {occ}.", "{subj}被{occ}部分遮住。"),
            ]
            occ = o_list[0]; subj_o = o_list[1]
            t = pick(OBJ_OCC_TEMPLATES, rng.randint(0, 99) % len(OBJ_OCC_TEMPLATES))
            add(t[0].format(subj=fmt_o(subj_o), occ=fmt_o(occ)),
                t[1].format(subj=fmt_ozh(subj_o), occ=fmt_ozh(occ)))

    # ── 句子优先级排序 ──────────────────────────────────────────────────────────
    # 1. 主遮挡句（human/object as subject, not "The X..."）：最高优先，防止被截断
    # 2. 普通句（无遮挡关键词）：正常顺序
    # 3. 物体-物体遮挡补充句（"The X is partly hidden by..."）：最低优先
    #    → 当 o[1] 已被 H-O 交互句覆盖时，此句变为 optional，可被 interaction 截断策略丢弃
    _OCC_KWS = ("hidden", "behind", "occlud", "blocked", "obscured", "blocking")
    primary_occ = [(e,z) for e,z in raw_sents if any(k in e.lower() for k in _OCC_KWS) and not e.startswith("The ")]
    object_occ  = [(e,z) for e,z in raw_sents if any(k in e.lower() for k in _OCC_KWS) and e.startswith("The ")]
    other_raw   = [(e,z) for e,z in raw_sents if not any(k in e.lower() for k in _OCC_KWS)]
    raw_sents = primary_occ + other_raw + object_occ

    # ── 标注每句引用的实体 ─────────────────────────────────────────────────────
    # 用 display name 在句子文本中搜索（full-string substring match）
    all_ids = list(selected_h) + list(selected_o)
    display_map = {(hn(e) if e in set(HUMANS) else on_(e)): e for e in all_ids}

    def find_refs(en_text):
        return frozenset(eid for dname, eid in display_map.items() if dname in en_text)

    sentences = [(en, zh, find_refs(en)) for en, zh in raw_sents]
    return sentences


# OO occlusion template banks for v12:
# - generic: broad safe phrasing
# - typed: preferred templates by (subj_type, occ_type)
# - ground: for small occluders at level=2
OO_OCC_GENERIC = [
    ("The {subj} is partly hidden by the {occ}.", "{subj}被{occ}部分遮挡。", "hidden_by"),
    ("The {subj} is partly obscured by the {occ}.", "{subj}被{occ}部分遮住。", "obscured_by"),
    ("The {occ} partly blocks the {subj}.", "{occ}部分挡住了{subj}。", "partly_blocks"),
    ("The {occ} partly covers the {subj}.", "{occ}部分遮住了{subj}。", "partly_covers"),
    ("The {subj} overlaps with the {occ}, with partial occlusion.", "{subj}与{occ}部分重叠并产生遮挡。", "overlaps"),
    ("The {subj} is in front of the {occ}, partially blocking it.", "{subj}位于{occ}前方，并部分挡住了它。", "in_front_blocking"),
]

OO_OCC_GROUND = [
    ("The {subj} is partly hidden by the {occ} on the ground.", "{subj}被地面上的{occ}部分遮挡。", "ground_hidden_by"),
    ("The {subj} is partly obscured by the {occ} on the ground.", "{subj}被地面上的{occ}部分遮住。", "ground_obscured_by"),
    ("The {occ} on the ground partly blocks the {subj}.", "地面上的{occ}部分挡住了{subj}。", "ground_partly_blocks"),
    ("The {occ} on the ground partly covers the {subj}.", "地面上的{occ}部分遮住了{subj}。", "ground_partly_covers"),
    ("The {subj} overlaps with the {occ} on the ground, with partial occlusion.", "{subj}与地面上的{occ}部分重叠并产生遮挡。", "ground_overlaps"),
]

OO_OCC_TYPED = {
    ("food", "object"): [
        ("The {occ} partly blocks the {subj}.", "{occ}部分挡住了{subj}。", "food_obj_blocks"),
        ("The {subj} is partly obscured by the {occ}.", "{subj}被{occ}部分遮住。", "food_obj_obscured"),
    ],
    ("object", "food"): [
        ("The {subj} is partly obscured by the {occ}.", "{subj}被{occ}部分遮住。", "obj_food_obscured"),
        ("The {occ} partly covers the {subj}.", "{occ}部分遮住了{subj}。", "obj_food_covers"),
    ],
    ("food", "food"): [
        ("The {subj} is partly hidden by the {occ}.", "{subj}被{occ}部分遮挡。", "food_food_hidden"),
        ("The {subj} overlaps with the {occ}, with partial occlusion.", "{subj}与{occ}部分重叠并产生遮挡。", "food_food_overlaps"),
    ],
    ("object", "object"): [
        ("The {occ} partly blocks the {subj}.", "{occ}部分挡住了{subj}。", "obj_obj_blocks"),
        ("The {subj} is partly obscured by the {occ}.", "{subj}被{occ}部分遮住。", "obj_obj_obscured"),
        ("The {subj} overlaps with the {occ}, with partial occlusion.", "{subj}与{occ}部分重叠并产生遮挡。", "obj_obj_overlaps"),
    ],
    ("animal", "object"): [
        ("The {animal} stands in front of the {occ}, partially blocking it.", "{animal}站在{occ}前方，并部分挡住了它。", "animal_obj_front_blocking"),
        ("The {animal} overlaps with the {occ}, with partial occlusion.", "{animal}与{occ}部分重叠并产生遮挡。", "animal_obj_overlaps"),
    ],
    ("object", "animal"): [
        ("The {occ} partly blocks the {obj}.", "{occ}部分挡住了{obj}。", "obj_animal_blocks"),
        ("The {obj} overlaps with the {occ}, with partial occlusion.", "{obj}与{occ}部分重叠并产生遮挡。", "obj_animal_overlaps"),
    ],
    ("animal", "food"): [
        ("The {animal} stands in front of the {occ}, partially blocking it.", "{animal}站在{occ}前方，并部分挡住了它。", "animal_food_front_blocking"),
        ("The {animal} overlaps with the {occ}, with partial occlusion.", "{animal}与{occ}部分重叠并产生遮挡。", "animal_food_overlaps"),
    ],
    ("food", "animal"): [
        ("The {occ} partly blocks the {food}.", "{occ}部分挡住了{food}。", "food_animal_blocks"),
        ("The {food} overlaps with the {occ}, with partial occlusion.", "{food}与{occ}部分重叠并产生遮挡。", "food_animal_overlaps"),
    ],
    ("animal", "animal"): [
        ("The {subj} stands in front of the {occ}, partially blocking it.", "{subj}站在{occ}前方，并部分挡住了它。", "animal_animal_front_blocking"),
        ("The {subj} overlaps with the {occ}, with partial occlusion.", "{subj}与{occ}部分重叠并产生遮挡。", "animal_animal_overlaps"),
    ],
}

# OOi typed templates with lightweight semantic compatibility.
# Use only when (subj, occ) pair satisfies the relation constraints.
OOI_TOP_SET = {
    "white_mug", "apple", "donut", "burger", "pizza_slice", "sushi_set",
    "headphones", "helmet", "dslr_camera", "table_lamp", "toolbox",
    "orange_basketball", "soccer_ball"
}
OOI_BASE_SET = {
    "wooden_dining_chair", "travel_suitcase", "toolbox", "red_backpack",
    "open_silver_laptop"
}
OOI_LEANABLE_SUBJ = {
    "folding_umbrella", "tennis_racket", "acoustic_guitar", "tripod",
    "skateboard", "watering_can"
}
OOI_LEANABLE_OCC = {
    "wooden_dining_chair", "travel_suitcase", "toolbox", "red_backpack",
    "blue_bicycle"
}
OOI_HANGABLE = {"headphones", "folding_umbrella", "tennis_racket"}
OOI_ATTACHABLE = {"headphones", "helmet", "table_lamp", "tripod"}
OOI_CLIPPABLE = {"headphones", "table_lamp"}

OOI_TEMPLATES = [
    ("The {subj} rests on the {occ}.", "{subj}放在{occ}上。", "rests_on"),
    ("The {subj} leans against the {occ}.", "{subj}倚靠在{occ}上。", "leans_against"),
    ("The {subj} hangs on the {occ}.", "{subj}挂在{occ}上。", "hangs_on"),
    ("The {subj} is attached to the {occ}.", "{subj}附着在{occ}上。", "attached_to"),
    ("The {subj} is strapped to the {occ}.", "{subj}绑在{occ}上。", "strapped_to"),
    ("The {subj} is tucked under the {occ}.", "{subj}塞在{occ}下面。", "tucked_under"),
    ("The {subj} is propped against the {occ}.", "{subj}斜靠在{occ}上。", "propped_against"),
    ("The {subj} is set atop the {occ}.", "{subj}放在{occ}上方。", "set_atop"),
    ("The {subj} clips onto the {occ}.", "{subj}夹在{occ}上。", "clips_onto"),
]

ANIMAL_ANIMAL_OOI_TEMPLATES = [
    ("The {subj} touches the {occ}.", "{subj}触碰着{occ}。", "touches"),
    ("The {subj} nudges the {occ}.", "{subj}轻碰着{occ}。", "nudges"),
    ("The {subj} nuzzles the {occ}.", "{subj}用鼻子轻蹭着{occ}。", "nuzzles"),
    ("The {subj} paws at the {occ}.", "{subj}用爪子拨弄着{occ}。", "paws_at"),
    ("The {subj} pecks at the {occ}.", "{subj}啄着{occ}。", "pecks_at"),
]
def _pick_for_pair_no_repeat(candidates, pair_key, rng, pair_tracker=None):
    """Pick template with per-pair non-repeat preference.
    candidates: [(en, zh, tmpl_key), ...]
    pair_key: usually (occ, subj)
    pair_tracker: dict[pair_key] -> set(tmpl_key)
    """
    if not candidates:
        return None
    if pair_tracker is None:
        c = candidates[rng.randint(0, 10_000) % len(candidates)]
        return c[0], c[1], c[2]

    used = pair_tracker.get(pair_key, set())
    fresh = [c for c in candidates if c[2] not in used]
    pick_pool = fresh if fresh else candidates
    c = pick_pool[rng.randint(0, 10_000) % len(pick_pool)]
    pair_tracker.setdefault(pair_key, set()).add(c[2])
    return c[0], c[1], c[2]


def ensure_entity_mentions(filtered, sel_h, sel_o):
    """Ensure every selected entity is explicitly mentioned at least once in prompt sentences."""
    out = list(filtered)
    text = " ".join(e for e, _ in out)

    def _has_entity(ent):
        return ent.replace("_", " ") in text

    # ensure human mentions
    for h in sel_h:
        if _has_entity(h):
            continue
        ref_o = sel_o[0] if sel_o else None
        ref_h = next((hh for hh in sel_h if hh != h), None)
        if ref_o:
            en = f"{hn(h)} stands near the {on_(ref_o)}."
            zh = f"{hn_zh(h)}站在{on_zh(ref_o)}附近。"
        elif ref_h:
            en = f"{hn(h)} stands beside {hn(ref_h)}."
            zh = f"{hn_zh(h)}站在{hn_zh(ref_h)}旁边。"
        else:
            en = f"{hn(h)} stands in the scene."
            zh = f"{hn_zh(h)}站在场景中。"
        out.append((en, zh))
        text += " " + en

    # ensure object mentions
    for o in sel_o:
        if _has_entity(o):
            continue
        ref_o = next((oo for oo in sel_o if oo != o), None)
        ref_h = sel_h[0] if sel_h else None
        if ref_o:
            en = f"The {on_(o)} is beside the {on_(ref_o)}."
            zh = f"{on_zh(o)}在{on_zh(ref_o)}旁边。"
        elif ref_h:
            en = f"{hn(ref_h)} stands near the {on_(o)}."
            zh = f"{hn_zh(ref_h)}站在{on_zh(o)}附近。"
        else:
            en = f"The {on_(o)} is visible."
            zh = f"{on_zh(o)}可见。"
        out.append((en, zh))
        text += " " + en

    return out




# ── 7. v12 采样辅助（去重 + 动词均衡） ───────────────────────────────────────
def split_sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def normalize_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _extract_verb_after_subject(sentence, subject_phrase):
    s = normalize_text(sentence)
    sp = subject_phrase.lower()
    if s.startswith(sp + " "):
        tail = s[len(sp) + 1 :]
    else:
        tail = s
    m = re.match(r"([a-z]+(?:\s+[a-z]+){0,3})\b", tail)
    return m.group(1).strip() if m else ""


def _extract_oo_occ_key(sentence):
    ns = normalize_text(sentence)
    checks = [
        ("ground_partly_blocks", "on the ground partly blocks"),
        ("ground_partly_covers", "on the ground partly covers"),
        ("ground_hidden_by", "on the ground"),
        ("ground_obscured_by", "on the ground"),
        ("partly_blocks", "partly blocks"),
        ("partly_covers", "partly covers"),
        ("obscured_by", "obscured by"),
        ("hidden_by", "hidden by"),
        ("in_front_blocking", "in front of"),
        ("overlaps", "overlaps"),
    ]
    for key, pat in checks:
        if pat in ns:
            return key
    return ""


def extract_units_and_verbs(prompt_en, sel_h, sel_o):
    """
    轻量级结构提取：
    - unit_key: 用于去重（HH/HO/OO 的主体-谓词-客体）
    - verbs: 用于均衡采样的动词集合
    """
    humans = {h: h.replace("_", " ").lower() for h in sel_h}
    objects = {o: o.replace("_", " ").lower() for o in sel_o}
    units = []
    verbs = []

    for sent in split_sents(prompt_en):
        ns = normalize_text(sent)
        if ns.startswith("white studio"):
            continue

        # subject 优先匹配 human，其次 object
        subj = None
        subj_type = None
        for h, hp in sorted(humans.items(), key=lambda kv: len(kv[1]), reverse=True):
            if ns.startswith(hp + " ") or ns == hp:
                subj, subj_type = h, "H"
                break
        if subj is None:
            for o, op in sorted(objects.items(), key=lambda kv: len(kv[1]), reverse=True):
                if ns.startswith("the " + op + " ") or ns.startswith(op + " "):
                    subj, subj_type = o, "O"
                    break

        if subj is None:
            continue

        # HH: 句中出现另一个 human
        if subj_type == "H":
            for h2, hp2 in humans.items():
                if h2 == subj:
                    continue
                if re.search(r"(?<![a-z])" + re.escape(hp2) + r"(?![a-z])", ns):
                    v = _extract_verb_after_subject(sent, humans[subj])
                    units.append(("HH", subj, v, h2))
                    if v:
                        verbs.append(v)
                    break

        # HO/OO: subject + the object
        for o, op in objects.items():
            if o == subj:
                continue
            if re.search(r"\bthe\s+" + re.escape(op) + r"\b", ns):
                v = _extract_verb_after_subject(sent, humans[subj] if subj_type == "H" else ("the " + objects[subj]))
                utype = "HO" if subj_type == "H" else "OO"
                units.append((utype, subj, v, o))
                if utype == "OO":
                    ov = _extract_oo_occ_key(sent)
                    if ov:
                        verbs.append(ov)
                    elif v:
                        verbs.append(v)
                elif v:
                    verbs.append(v)
                break

    unit_key = tuple(sorted(units))
    verb_key = tuple(sorted(verbs))
    return unit_key, verb_key


def _neutral_h_sentence(h):
    return f"{hn(h)} is also in the scene.", f"{hn_zh(h)}也出现在画面中。"


def _neutral_o_sentence(o):
    return f"The {on_(o)} is also visible.", f"{on_zh(o)}也出现在画面中。"


def _pop_aoi_pair(available_o):
    animals = [o for o in available_o if entity_type(o) == "animal"]
    non_animals = [o for o in available_o if entity_type(o) != "animal"]
    if animals and non_animals:
        a = animals[0]
        b = non_animals[0]
        available_o.remove(a)
        available_o.remove(b)
        return [a, b]
    return None


def _allocate_slots_for_level(tag, ratio, level, sel_h, sel_o):
    plan = SLOT_PLAN[(ratio, tag)][level]
    available_h = list(sel_h)
    available_o = list(sel_o)
    slots = []

    # HO first: consumes both a human and an object.
    for cat in ("HOs", "HOocc", "HOi"):
        need = plan.get(cat, 0)
        for _ in range(need):
            if not available_h or not available_o:
                break
            h = available_h.pop(0)
            o = available_o.pop(0)
            slots.append((cat, [h], [o]))

    # HH next: use remaining humans in disjoint pairs.
    for cat in ("HHs", "HHocc", "HHi"):
        need = plan.get(cat, 0)
        for _ in range(need):
            if len(available_h) < 2:
                break
            h1 = available_h.pop(0)
            h2 = available_h.pop(0)
            slots.append((cat, [h1, h2], []))

    # Non-human interaction slots: prefer AOi, otherwise fall back to OOi as part of the same structural slot.
    need_noi = plan.get("NOi", 0)
    for _ in range(need_noi):
        pair = _pop_aoi_pair(available_o)
        if pair is None:
            if len(available_o) >= 2:
                pair = [available_o.pop(0), available_o.pop(0)]
                slots.append(("NOi", [], pair))
                continue
            break
        slots.append(("NOi", [], pair))

    for cat in ("OOs", "OOocc", "OOi"):
        need = plan.get(cat, 0)
        for _ in range(need):
            if len(available_o) < 2:
                break
            o1 = available_o.pop(0)
            o2 = available_o.pop(0)
            slots.append((cat, [], [o1, o2]))

    return slots


def _select_slots_from_full(full_slots, target_plan):
    kept = []
    by_cat = defaultdict(list)
    for slot in full_slots:
        by_cat[slot[0]].append(slot)
    for cat in REL_CATS:
        need = target_plan.get(cat, 0)
        if need:
            kept.extend(by_cat.get(cat, [])[:need])
    return kept


def _entities_from_slots(slots, all_h, all_o):
    used_h = set()
    used_o = set()
    for _, hs, os_ in slots:
        used_h.update(hs)
        used_o.update(os_)
    sel_h = [h for h in all_h if h in used_h]
    sel_o = [o for o in all_o if o in used_o]
    return sel_h, sel_o


def _expand_entities_to_level(slots, all_h, all_o, target_n_h, target_n_o):
    slot_h, slot_o = _entities_from_slots(slots, all_h, all_o)
    sel_h = list(slot_h)
    sel_o = list(slot_o)
    for h in all_h:
        if len(sel_h) >= target_n_h:
            break
        if h not in sel_h:
            sel_h.append(h)
    for o in all_o:
        if len(sel_o) >= target_n_o:
            break
        if o not in sel_o:
            sel_o.append(o)
    return sel_h, sel_o


def _slot_signature(slot):
    cat, hs, os_ = slot
    return (cat, tuple(hs), tuple(os_))


def _ensure_sentence_punct(en, zh):
    en = en.strip()
    zh = zh.strip()
    if en and en[-1] not in ".!?":
        en += "."
    if zh and zh[-1] not in "。！？":
        zh += "。"
    return en, zh


def _realize_full_slots(current_seed, sel_h, sel_o, full_slots):
    rng = random.Random(current_seed * 7919 + 8 * 97 + 13)
    realized = []
    seen_en = set()
    seen_sig = set()
    for i, (cat, hs, os_) in enumerate(full_slots):
        built = _build_relation_sentence(cat, hs, os_, rng, 8, None, variant_idx=i)
        if not built:
            continue
        en, zh = built
        en, zh = _ensure_sentence_punct(en, zh)
        sig = _relation_signature(en, sel_h, sel_o)
        if sig is None:
            sig = ("TEXT", en)
        if en in seen_en or sig in seen_sig:
            continue
        realized.append((_slot_signature((cat, hs, os_)), en, zh))
        seen_en.add(en)
        seen_sig.add(sig)
    return realized


def _render_selected_slots(selected_slots, full_realized, sel_h, sel_o):
    wanted = {_slot_signature(s) for s in selected_slots}
    selected = [(en, zh) for sig, en, zh in full_realized if sig in wanted]
    covered_text = " ".join(en for en, _ in selected)
    for h in sel_h:
        if h.replace("_", " ") not in covered_text:
            en, zh = _neutral_h_sentence(h)
            en, zh = _ensure_sentence_punct(en, zh)
            selected.append((en, zh))
            covered_text += " " + en
    for o in sel_o:
        if o.replace("_", " ") not in covered_text:
            en, zh = _neutral_o_sentence(o)
            en, zh = _ensure_sentence_punct(en, zh)
            selected.append((en, zh))
            covered_text += " " + en
    return selected


def build_entries_for_seed(current_tag, current_ratio, current_seed, lv2_pair_template_tracker):
    """v13: allocate entity-disjoint core relation slots first, then realize them."""
    rng = random.Random(current_seed)
    max_n_h, max_n_o = RATIO_MAP[current_ratio][8]

    h_pool = rng.sample(HUMANS, k=len(HUMANS))
    o_pool = rng.sample(NON_HUMANS, k=len(NON_HUMANS))

    # Keep object prefixes semantically aligned with the target bucket.
    if current_tag == "occlusion_interaction" and current_ratio == "object_heavy":
        animal_pos = next((i for i, x in enumerate(o_pool) if entity_type(x) == "animal"), None)
        non_animal_pos = next((i for i, x in enumerate(o_pool) if entity_type(x) != "animal"), None)
        if animal_pos is not None:
            o_pool[0], o_pool[animal_pos] = o_pool[animal_pos], o_pool[0]
        if non_animal_pos is not None:
            non_animal_pos = next((i for i, x in enumerate(o_pool) if i != 0 and entity_type(x) != "animal"), None)
            if non_animal_pos is not None:
                o_pool[1], o_pool[non_animal_pos] = o_pool[non_animal_pos], o_pool[1]

    all_h = h_pool[:max_n_h]
    all_o = o_pool[:max_n_o]
    full_slots = _allocate_slots_for_level(current_tag, current_ratio, 8, all_h, all_o)
    full_realized = _realize_full_slots(current_seed, all_h, all_o, full_slots)

    drafts = []
    for lv in [8, 6, 4, 2]:
        target_plan = SLOT_PLAN[(current_ratio, current_tag)][lv]
        slots = _select_slots_from_full(full_slots, target_plan)
        target_n_h, target_n_o = RATIO_MAP[current_ratio][lv]
        sel_h, sel_o = _expand_entities_to_level(slots, all_h, all_o, target_n_h, target_n_o)
        n_h, n_o = target_n_h, target_n_o
        realized = _render_selected_slots(slots, full_realized, sel_h, sel_o)
        en = "White studio. Keep all entities at realistic scale. " + " ".join(e for e, _ in realized)
        zh = "白色摄影棚。保持所有实体为真实比例。" + "".join(z for _, z in realized)

        unit_key, verb_key = extract_units_and_verbs(en, sel_h, sel_o)
        drafts.append({
            "seed_id": current_seed,
            "level": lv,
            "class_tag": current_tag,
            "ratio_type": current_ratio,
            "n_humans": n_h,
            "n_objects": n_o,
            "total_entities": lv,
            "people_names": sel_h,
            "object_names": sel_o,
            "prompt_en": en,
            "prompt_zh": zh,
            "token_len_est": est_tokens(en),
            "_unit_key": unit_key,
            "_verb_key": verb_key,
        })
    return drafts


def _allocate_combo_targets(num_seeds, combos):
    base = num_seeds // len(combos)
    rem = num_seeds % len(combos)
    out = {}
    for i, c in enumerate(combos):
        out[c] = base + (1 if i < rem else 0)
    return out


def _pick_template(lst, rng):
    return lst[rng.randint(0, 10**9) % len(lst)]


def _pick_filtered_template(lst, rng, allow_substrings=None, deny_substrings=None):
    allow_substrings = allow_substrings or []
    deny_substrings = deny_substrings or []
    cand = []
    for t in lst:
        en = t[0].lower()
        if allow_substrings and not any(s in en for s in allow_substrings):
            continue
        if any(s in en for s in deny_substrings):
            continue
        cand.append(t)
    if cand:
        return _pick_template(cand, rng)
    return _pick_template(lst, rng) if lst else None


def _with_template_key(templates):
    out = []
    for en, zh in templates:
        key = normalize_text(en)
        key = key.replace("{o}", "").replace("{obj}", "").replace("{f}", "")
        key = re.sub(r"\s+", " ", key).strip(" .")
        out.append((en, zh, key))
    return out

def _h(name): return hn(name)
def _hz(name): return hn_zh(name)
def _o(name): return on_(name)
def _oz(name): return on_zh(name)


def _front_reorder(seq, front_idx):
    if not seq:
        return list(seq)
    i = front_idx % len(seq)
    return [seq[i]] + [x for j, x in enumerate(seq) if j != i]


def _front_reorder_pair(seq, pair_idx, ordered=True):
    n = len(seq)
    if n < 2:
        return list(seq)
    if ordered:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    i, j = pairs[pair_idx % len(pairs)]
    out = [seq[i], seq[j]]
    for k, x in enumerate(seq):
        if k not in (i, j):
            out.append(x)
    return out


def _cat_order_variant(cat, sel_h, sel_o, variant_idx):
    humans = list(sel_h)
    objs = list(sel_o)
    if variant_idx is None:
        return humans, objs

    if cat in ("HHs", "HHocc", "HHi") and len(humans) >= 2:
        humans = _front_reorder_pair(humans, variant_idx, ordered=True)
    elif cat in ("HOs", "HOocc", "HOi") and humans and objs:
        # Diagonal-style sweep so early variants change both human and object.
        hi = variant_idx % len(humans)
        oi = variant_idx % len(objs)
        humans = _front_reorder(humans, hi)
        objs = _front_reorder(objs, oi)
    elif cat in ("OOs", "OOocc", "OOi") and len(objs) >= 2:
        objs = _front_reorder_pair(objs, variant_idx, ordered=True)
    elif cat == "AOi" and objs:
        objs = _front_reorder(objs, variant_idx)
    return humans, objs


def _oo_occ_template_pool(subj, occ, level):
    subj_t = entity_type(subj)
    occ_t = entity_type(occ)
    typed = list(OO_OCC_TYPED.get((subj_t, occ_t), []))
    generic = list(OO_OCC_GENERIC)
    if occ not in VALID_OCCLUDERS:
        # Global rule: small occluders use ground phrasing.
        return OO_OCC_GROUND + typed + generic
    return typed + generic


def _ooi_template_pool(subj, occ):
    pool = []
    subj_t = entity_type(subj)
    occ_t = entity_type(occ)

    if subj_t == "animal" and occ_t == "animal":
        pool.extend(ANIMAL_ANIMAL_OOI_TEMPLATES[:2])  # touches, nudges
        if subj in NUZZLE_ANIMALS:
            pool.append(ANIMAL_ANIMAL_OOI_TEMPLATES[2])  # nuzzles
        if subj in PAW_ANIMALS:
            pool.append(ANIMAL_ANIMAL_OOI_TEMPLATES[3])  # paws_at
        if subj in BIRD_ANIMALS:
            pool.append(ANIMAL_ANIMAL_OOI_TEMPLATES[4])  # pecks_at
    # Safe default for object-object pairs.
    elif subj_t == "object" and occ_t == "object":
        pool.append(OOI_TEMPLATES[0])  # rests_on
        if subj in OOI_LEANABLE_SUBJ and occ in OOI_LEANABLE_OCC:
            pool.extend([OOI_TEMPLATES[1], OOI_TEMPLATES[6]])  # leans/propped
        if subj in OOI_HANGABLE and occ in OOI_BASE_SET:
            pool.append(OOI_TEMPLATES[2])  # hangs_on
        if subj in OOI_ATTACHABLE and occ in OOI_BASE_SET:
            pool.extend([OOI_TEMPLATES[3], OOI_TEMPLATES[4]])  # attached/strapped
        if subj in OOI_TOP_SET and occ in OOI_BASE_SET:
            pool.append(OOI_TEMPLATES[7])  # set_atop
        if subj in OOI_CLIPPABLE and occ in OOI_BASE_SET:
            pool.append(OOI_TEMPLATES[8])  # clips_onto
        if subj in OOI_TOP_SET and occ in OOI_BASE_SET:
            pool.append(OOI_TEMPLATES[5])  # tucked_under
    else:
        # For pairs involving animals/foods, keep conservative non-contact-ish support relation.
        pool.append(OOI_TEMPLATES[0])  # rests_on

    # De-dup preserving order
    seen = set()
    out = []
    for t in pool:
        if t[2] in seen:
            continue
        seen.add(t[2])
        out.append(t)
    return out if out else [OOI_TEMPLATES[0]]


def _aoi_l2_suffix(rng, enable):
    if not enable:
        return "", ""
    variants = [
        ("", ""),
        (" with partial overlap.", "，画面有部分重叠。"),
        (" while partly blocking it.", "，并部分挡住了它。"),
        (" in the foreground with overlap.", "，并在前景形成重叠。"),
        (" with visible occlusion.", "，可见遮挡关系。"),
    ]
    return _pick_template(variants, rng)


def _cat_sentence(sent, sel_h, sel_o):
    ns = normalize_text(sent)
    human_ph = {h: h.replace("_", " ").lower() for h in sel_h}
    obj_ph = {o: o.replace("_", " ").lower() for o in sel_o}
    h_present = [h for h, ph in human_ph.items() if re.search(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])", ns)]
    o_present = [o for o, ph in obj_ph.items() if re.search(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])", ns)]
    occ = any(k in ns for k in ("hidden", "obscured", "blocking", "occlud", "partly blocks", "partly covers", "overlaps"))

    if len(set(h_present)) >= 2:
        if occ:
            return "HHocc"
        if any(v in ns for v in ("high-fives", "fist-bumps", "hugs", "holds hands with", "shakes hands with",
                                 "taps", "pats", "links arms with", "walks arm in arm with",
                                 "places a hand on", "pulls", "leans on")):
            return "HHi"
        return "HHs"

    if len(h_present) >= 1 and len(o_present) >= 1:
        if occ:
            return "HOocc"
        # static-ish verbs
        if any(v in ns for v in ("stands near", "stands beside", "looks at", "faces", "gazes at", "observes",
                                 "glances at", "approaches", "steps toward", "leans toward",
                                 "pauses beside")):
            return "HOs"
        return "HOi"

    # AOi first: animal subject interacting with food/object
    if len(o_present) >= 2:
        subj_animal = False
        for an in ANIMALS:
            anp = an.replace("_", " ")
            if ns.startswith("the " + anp + " ") or ns.startswith(anp + " "):
                subj_animal = True
                break
        if subj_animal and any(v in ns for v in ("nibbles", "sniffs", "paws at", "licks", "pecks at", "chews", "nudges", "grips", "perches on", "sits on", "curls up on", "lifts", "carries", "holds")):
            return "AOi"
        if occ:
            return "OOocc"
        return "OOi" if any(v in ns for v in ("rests on", "leans against", "hang on", "hangs on", "attached to", "strapped to", "tucked under", "propped against", "set atop", "clips onto")) else "OOs"
    return None


def _present_entities(sent, sel_h, sel_o):
    ns = normalize_text(sent)
    human_ph = [(h, h.replace("_", " ").lower()) for h in sel_h]
    obj_ph = [(o, o.replace("_", " ").lower()) for o in sel_o]
    h_present = [h for h, ph in human_ph if re.search(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])", ns)]
    o_present = [o for o, ph in obj_ph if re.search(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])", ns)]
    return h_present, o_present


def _relation_signature(sent, sel_h, sel_o):
    cat = _cat_sentence(sent, sel_h, sel_o)
    if not cat:
        return None
    h_present, o_present = _present_entities(sent, sel_h, sel_o)
    if cat in ("HOs", "HOi", "HOocc") and len(h_present) >= 1 and len(o_present) >= 1:
        return (cat, h_present[0], o_present[0])
    if cat in ("HHs", "HHi", "HHocc") and len(h_present) >= 2:
        return (cat, h_present[0], h_present[1])
    if cat in ("OOs", "OOi", "OOocc", "AOi", "NOi") and len(o_present) >= 2:
        return (cat, o_present[0], o_present[1])
    return None


def _build_relation_sentence(cat, sel_h, sel_o, rng, level, level2_verb_tracker=None, variant_idx=None):
    humans, objs = _cat_order_variant(cat, sel_h, sel_o, variant_idx)
    animals = [o for o in objs if entity_type(o) == "animal"]
    foods = [o for o in objs if entity_type(o) == "food"]
    objects = [o for o in objs if entity_type(o) == "object"]

    if cat == "HHi" and len(humans) >= 2:
        h1, h2 = humans[0], humans[1]
        t = _pick_template(HH_INTERACTION_TEMPLATES, rng)
        return t[0].format(h1=_h(h1), h2=_h(h2)), t[1].format(h1=_hz(h1), h2=_hz(h2))
    if cat == "HHs" and len(humans) >= 2:
        h1, h2 = humans[0], humans[1]
        t = _pick_template(HH_STATIC_TEMPLATES, rng)
        return t[0].format(h1=_h(h1), h2=_h(h2)), t[1].format(h1=_hz(h1), h2=_hz(h2))
    if cat == "HHocc" and len(humans) >= 2:
        h1, h2 = humans[0], humans[1]
        t = _pick_template(HH_OCC_TEMPLATES, rng)
        return t[0].format(h1=_h(h1), h2=_h(h2)), t[1].format(h1=_hz(h1), h2=_hz(h2))
    if cat == "HOs" and len(humans) >= 1 and len(objs) >= 1:
        h, o = humans[0], objs[0]
        t = _pick_template(STATIC_H_ENTITY, rng)
        return t[0].format(h=_h(h), o=_o(o)), t[1].format(h=_hz(h), o=_oz(o))
    if cat == "HOocc" and len(humans) >= 1 and len(objs) >= 1:
        h, o = humans[0], objs[0]
        allow_small_ground = True
        occ_en, occ_zh = occ_desc(o, allow_small_ground)
        if o in VALID_OCCLUDERS:
            pool = list(OCCLUSION_TEMPLATES) + list(HO_OCC_LARGE_PARTIAL)
        else:
            pool = list(OCCLUSION_TEMPLATES) + list(HO_OCC_SMALL_GROUND)
        t = _pick_template(pool, rng)
        return t[0].format(subj=_h(h), occ=occ_en), t[1].format(subj=_hz(h), occ=occ_zh)
    if cat == "HOi" and len(humans) >= 1 and len(objs) >= 1:
        h, o = humans[0], objs[0]
        ot = entity_type(o)
        if ot == "food":
            t = _pick_filtered_template(
                H_FOOD.get(o, [("{h} holds the {o}.", "{h}拿着{o}。")]),
                rng,
                allow_substrings=["holds", "eats", "offers", "slices", "tears", "bites", "raises", "carries", "polishes", "smells"],
            )
        elif ot == "animal":
            t = _pick_filtered_template(
                H_ANIMAL.get(o, [("{h} pets the {o}.", "{h}抚摸着{o}。")]),
                rng,
                deny_substrings=[
                    "reaches toward", "looks at", "glances at", "faces",
                    "approaches", "steps toward", "leans toward",
                    "watches", "walks beside", "crouches beside", "gestures toward",
                ],
            )
        else:
            t = _pick_filtered_template(
                H_OBJECT.get(o, [("{h} carries the {o}.", "{h}拿着{o}。")]),
                rng,
                deny_substrings=[
                    "reaches toward", "looks at", "glances at", "faces",
                    "approaches", "steps toward", "leans toward",
                    "stands near", "stands beside", "in front of", "pauses beside",
                ],
            )
        return t[0].format(h=_h(h), o=_o(o)), t[1].format(h=_hz(h), o=_oz(o))
    if cat == "OOs" and len(objs) >= 2:
        a, b = objs[0], objs[1]
        en_tpl, zh_tpl, _ = _pick_template(OO_STATIC_TEMPLATES, rng)
        return en_tpl.format(subj=_o(a), occ=_o(b)), zh_tpl.format(subj=_oz(a), occ=_oz(b))
    if cat == "OOocc" and len(objs) >= 2:
        a, b = objs[0], objs[1]
        pool = _oo_occ_template_pool(a, b, level)
        t = _pick_template(pool, rng) if pool else None
        if t:
            en_tpl, zh_tpl, _ = t
            fmt = {
                "subj": _o(a),
                "occ": _o(b),
                "animal": _o(a) if entity_type(a) == "animal" else _o(b),
                "obj": _o(a) if entity_type(a) == "object" else _o(b),
                "food": _o(a) if entity_type(a) == "food" else _o(b),
            }
            fmt_zh = {
                "subj": _oz(a),
                "occ": _oz(b),
                "animal": _oz(a) if entity_type(a) == "animal" else _oz(b),
                "obj": _oz(a) if entity_type(a) == "object" else _oz(b),
                "food": _oz(a) if entity_type(a) == "food" else _oz(b),
            }
            return en_tpl.format(**fmt), zh_tpl.format(**fmt_zh)
        # safe fallback
        allow_small_ground = True
        occ_en, occ_zh = occ_desc(b, allow_small_ground)
        return f"The {_o(a)} is partly hidden by the {occ_en}.", f"{_oz(a)}被{occ_zh}部分遮挡。"
    if cat in ("AOi", "NOi"):
        l2_oh_oi = (level == 2 and len(humans) == 0 and len(objs) == 2)
        if animals and foods:
            a, f = animals[0], foods[0]
            cands = A_FOOD.get(a, [("{o} sniffs the {f}.", "{o}嗅着{f}。")])
            cands = [t for t in cands if any(s in t[0].lower() for s in ["nibbles", "sniffs", "paws at", "licks", "pecks at", "chews", "lifts", "holds"])]
            if not cands:
                return None
            if level == 2 and level2_verb_tracker is not None:
                t = _pick_for_pair_no_repeat(_with_template_key(cands), ("AOi_food", a, f), rng, level2_verb_tracker)
                en_tpl, zh_tpl = t[0], t[1]
            else:
                en_tpl, zh_tpl = _pick_template(cands, rng)
            en = en_tpl.format(o=_o(a), f=_o(f)).rstrip(".")
            zh = zh_tpl.format(o=_oz(a), f=_oz(f)).rstrip("。")
            en_suf, zh_suf = _aoi_l2_suffix(rng, l2_oh_oi)
            return en + en_suf + ".", zh + zh_suf
        if animals and objects:
            a, ob = animals[0], objects[0]
            cands = A_OBJECT.get(a, [("{o} paws at the {obj}.", "{o}用爪子拨弄{obj}。")])
            cands = [t for t in cands if any(s in t[0].lower() for s in ["paws at", "nudges", "sniffs", "pecks at", "perches on", "grips", "carries", "sits on", "curls up on"])]
            if not cands:
                return None
            if level == 2 and level2_verb_tracker is not None:
                t = _pick_for_pair_no_repeat(_with_template_key(cands), ("AOi_obj", a, ob), rng, level2_verb_tracker)
                en_tpl, zh_tpl = t[0], t[1]
            else:
                en_tpl, zh_tpl = _pick_template(cands, rng)
            en = en_tpl.format(o=_o(a), obj=_o(ob)).rstrip(".")
            zh = zh_tpl.format(o=_oz(a), obj=_oz(ob)).rstrip("。")
            en_suf, zh_suf = _aoi_l2_suffix(rng, l2_oh_oi)
            return en + en_suf + ".", zh + zh_suf
        if len(objs) >= 2:
            a, b = objs[0], objs[1]
            pool = _ooi_template_pool(a, b)
            t = _pick_template(pool, rng) if pool else None
            if t:
                en_tpl, zh_tpl, _ = t
                return en_tpl.format(subj=_o(a), occ=_o(b)), zh_tpl.format(subj=_oz(a), occ=_oz(b))
            return f"The {_o(a)} rests on the {_o(b)}.", f"{_oz(a)}放在{_oz(b)}上。"
    if cat == "OOi" and len(objects) >= 2:
        a, b = objects[0], objects[1]
        pool = _ooi_template_pool(a, b)
        t = _pick_template(pool, rng) if pool else None
        if t:
            en_tpl, zh_tpl, _ = t
            return en_tpl.format(subj=_o(a), occ=_o(b)), zh_tpl.format(subj=_oz(a), occ=_oz(b))
        return f"The {_o(a)} rests on the {_o(b)}.", f"{_oz(a)}放在{_oz(b)}上。"
    return None


def enforce_relation_plan(tag, ratio, level, sel_h, sel_o, current_seed, filtered_pairs, level2_verb_tracker=None):
    plan = RELATION_PLAN[(ratio, tag)][level]
    rng = random.Random(current_seed * 7919 + level * 97)
    by_cat = defaultdict(list)
    for en, zh in filtered_pairs:
        c = _cat_sentence(en, sel_h, sel_o)
        if c:
            by_cat[c].append((en, zh))

    selected = []
    selected_en = set()
    selected_sig = set()
    for cat, n in plan.items():
        have = by_cat.get(cat, [])
        rng.shuffle(have)

        # de-dup pre-existing pairs in this category by relation signature,
        # not just exact sentence text, so the same entity pair won't appear twice
        # with slightly different wording.
        uniq_have = []
        seen_have_sig = set()
        for en, zh in have:
            sig = _relation_signature(en, sel_h, sel_o)
            if sig is None:
                sig = ("TEXT", en)
            if sig in seen_have_sig:
                continue
            seen_have_sig.add(sig)
            uniq_have.append((en, zh, sig))

        for en, zh, sig in uniq_have:
            if len([1 for x in selected if _cat_sentence(x[0], sel_h, sel_o) == cat]) >= n:
                break
            if en in selected_en:
                continue
            if sig in selected_sig:
                continue
            selected.append((en, zh))
            selected_en.add(en)
            selected_sig.add(sig)

        current_n = len([1 for x in selected if _cat_sentence(x[0], sel_h, sel_o) == cat])
        missing = n - current_n
        tries = 0
        variant_base = len(uniq_have)
        while missing > 0 and tries < max(12, n * 8):
            built = _build_relation_sentence(
                cat,
                sel_h,
                sel_o,
                rng,
                level,
                level2_verb_tracker,
                variant_idx=variant_base + tries,
            )
            tries += 1
            if not built:
                continue
            en, zh = built
            sig = _relation_signature(en, sel_h, sel_o)
            if sig is None:
                sig = ("TEXT", en)
            if en in selected_en:
                continue
            if sig in selected_sig:
                continue
            selected.append((en, zh))
            selected_en.add(en)
            selected_sig.add(sig)
            missing -= 1

    # optional enrichments kept minimal; keep deterministic and compact
    selected = ensure_entity_mentions(selected, sel_h, sel_o)
    return selected


def evaluate_entity_quality(prompt_en, sel_h, sel_o):
    text = normalize_text(prompt_en)
    required = [x.replace("_", " ").lower() for x in (list(sel_h) + list(sel_o))]
    all_names = [x.replace("_", " ").lower() for x in (HUMANS + NON_HUMANS)]
    required_set = set(required)
    covered = {n for n in required if re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", text)}
    extras = []
    for n in all_names:
        if n in required_set:
            continue
        if re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", text):
            extras.append(n)
    return len(covered) == len(required), len(extras) == 0, covered, extras


# ── 7. 主生成函数 ─────────────────────────────────────────────────────────────
def run_generation():
    tags = ["no_interaction_no_occlusion", "occlusion_no_interaction", "occlusion_interaction"]
    ratios = ["balanced", "human_heavy", "object_heavy"]
    combos = [(t, r) for t in tags for r in ratios]
    version = os.environ.get("DATASET_VERSION", "v13")

    file_configs = [
        (f"train_60k_{version}.jsonl", 60000),
        (f"val_1.5k_{version}.jsonl", 1500),
        (f"test_2.5k_{version}.jsonl", 2500),
        (f"extra_6k_{version}.jsonl", 6000),
    ]
    if os.environ.get("V12_PREVIEW") == "1":
        file_configs = [(f"preview_900_{version}.jsonl", 900)]

    global_id = 1
    seed_cursor = 20000
    max_tries = 18

    for filename, count in file_configs:
        print(f"正在生成 {filename}...")
        num_seeds = count // 4
        combo_targets = _allocate_combo_targets(num_seeds, combos)
        lv2_pair_template_tracker = {}
        quality = Counter()

        # 去重键与动词计数按 (tag,ratio,level) 维护
        used_keys = defaultdict(set)
        verb_counts = defaultdict(Counter)

        with open(filename, "w", encoding="utf-8") as f:
            for current_tag, current_ratio in combos:
                need = combo_targets[(current_tag, current_ratio)]
                produced = 0

                while produced < need:
                    best = None
                    fallback = None

                    for _ in range(max_tries):
                        seed = seed_cursor
                        seed_cursor += 1
                        drafts = build_entries_for_seed(
                            current_tag, current_ratio, seed, lv2_pair_template_tracker
                        )
                        if fallback is None:
                            fallback = drafts

                        # 去重检查（4 个 level 都尽量唯一）
                        dup_levels = []
                        score = 0.0
                        for d in drafts:
                            bkey = (current_tag, current_ratio, d["level"])
                            if d["_unit_key"] in used_keys[bkey]:
                                dup_levels.append(d["level"])
                                score += 1e6
                            # 动词均衡评分：已高频动词惩罚
                            vc = verb_counts[bkey]
                            for v in d["_verb_key"]:
                                score += vc[v]

                        if best is None or score < best[0]:
                            best = (score, drafts, dup_levels)
                        if not dup_levels:
                            break

                    chosen = best[1] if best is not None else fallback
                    # 提交写入，并更新去重/动词统计
                    for d in chosen:
                        bkey = (current_tag, current_ratio, d["level"])
                        used_keys[bkey].add(d["_unit_key"])
                        for v in d["_verb_key"]:
                            verb_counts[bkey][v] += 1
                        # Keep output field order consistent with historical files.
                        out = {
                            "id": global_id,
                            "seed_id": d["seed_id"],
                            "level": d["level"],
                            "class_tag": d["class_tag"],
                            "ratio_type": d["ratio_type"],
                            "n_humans": d["n_humans"],
                            "n_objects": d["n_objects"],
                            "total_entities": d["total_entities"],
                            "people_names": d["people_names"],
                            "object_names": d["object_names"],
                            "prompt_en": d["prompt_en"],
                            "prompt_zh": d["prompt_zh"],
                            "token_len_est": d["token_len_est"],
                        }
                        ok_cov, ok_extra, _, extras = evaluate_entity_quality(
                            out["prompt_en"], out["people_names"], out["object_names"]
                        )
                        quality["rows"] += 1
                        if ok_cov:
                            quality["entity_covered_rows"] += 1
                        if ok_extra:
                            quality["no_extra_entity_rows"] += 1
                        else:
                            quality["rows_with_extra_entity"] += 1
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        global_id += 1
                    produced += 1

        cov = (quality["entity_covered_rows"] / quality["rows"] * 100) if quality["rows"] else 0.0
        no_extra = (quality["no_extra_entity_rows"] / quality["rows"] * 100) if quality["rows"] else 0.0
        print(
            f"  完成: {filename} | seeds={num_seeds} | combos={len(combos)} "
            f"| entity_coverage={cov:.2f}% | no_extra_entity={no_extra:.2f}%"
        )

    print("--- v12 所有文件已生成完毕 ---")


if __name__ == "__main__":
    run_generation()
