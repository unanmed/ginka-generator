import { writeFile } from 'fs/promises';
import { autoLabelTowers } from './auto/auto';
import { IAutoLabelConfig, TowerColor } from './auto/types';
import { GinkaDataset, GinkaTrainData } from './types';

const [, , output, towerInfo, ...folders] = process.argv;

// 根据结果手动屏蔽一些满足条件但是不适合的塔
const ignoredTower: string[] = [
    'blossom',
    'cxzl_imprisoned',
    'hpxwzwy',
    'jianchuan1',
    'MC2D',
    'minecraft',
    'waterfultowerx',
    'xieyuhouxu',
    'cjst',
    'bdcxver1',
    'Curse_Prison',
    'DevilReturn',
    'forglory',
    'bdcxver1',
    'bleakst',
    'bucket',
    'callofxx',
    'CatLegend',
    'CatLegend2',
    'CatTowerAdven',
    'cjst',
    'Curse_Prison',
    'dfmzj',
    'Domains',
    'dreamofstar',
    'EmperorY_BA_1',
    'EpicCloudFantasyII',
    'evernight',
    'Finale_Fairy',
    'findaway',
    'fuzhaimota',
    'general',
    'guidaqiang',
    'guita',
    'H6Template',
    'HappyBirthday',
    'intentionaltower',
    'jianchuan2',
    'jiandanxiaota',
    'jiujiujiumota2',
    'Khronos',
    'kljhz',
    'LaWarden1',
    'levelstower',
    'lovehk01',
    'lycyjcc',
    'maplestoryx',
    'mdef2',
    'mengsi',
    'minglingfuchou',
    'mogai24',
    'mojiechuanqi',
    'moshenzhanzhenger',
    'moshenzhanzhengsan',
    'mota40cengnew',
    'MotaKurongge',
    'motatiangong',
    'MSadventure',
    'new_51_Mota_by_Nore',
    'night',
    'Nouveau50',
    'Nouveau50quaternaire',
    'Nouveau50second',
    'Nouveau50tertiaire',
    'OneStepToHellMOD',
    'princess_of_war_2',
    'revenge2',
    'revenge',
    'sengling',
    'SimpleMT',
    'srcs1',
    'stealing',
    'The_MT_of_Lulu_Farea',
    'tjrm1',
    'tyzt',
    'wanjiucc',
    'wuxingxianlu',
    'wuxingxianlu2',
    'wuxingxianlu3',
    'wxxl3mod',
    'x2xxj',
    'xdt',
    'xiangyaochumo',
    'xinxin2',
    'xinxin2new',
    'xyzywztt',
    'YAMT',
    'Yaya',
    'YCJQG4',
    'ygotower',
    'yiwangwuqian',
    'zd1',
    'ZEROTWO',
    'zjhy1',
    'zrsz',
    'zuozuozhieX_renewed',
    'ZZZE50Ex'
];

const ignoredFloor: Record<string, string[]> = {
    cxzl2: [
        'MN5',
        'MN7',
        'MN8',
        'MN49',
        'MN57',
        'MQ77',
        'MT85',
        'MZ75',
        'MT71',
        'ML26',
        'ML27',
        'MQ28'
    ],
    cxzl: [
        'MT63',
        'MT186',
        'MT256',
        'MT304',
        'MT313',
        'MT314',
        'MT334',
        'MT700',
        'MX92',
        'MT524',
        'MT646',
        'MT777',
        'MT296',
        'MT564'
    ],
    gl: ['MH2', 'MT21', 'YJ14'],
    gooTower2: ['MT96'],
    levelstower: ['LV0c'],
    lzlltete: ['MT0'],
    myslmdrydtwo: ['MT49'],
    OneStepToHell: ['MT02', 'MT05', 'MT07', 'MT09', 'MT44'],
    ousha: ['MT152', 'MT158', 'MT160'],
    Pacificwar: ['MT102', 'MT131'],
    tiandijinghz: ['MT54', 'MT41', 'A21', 'MT50', 'MT110', 'MT60'],
    wdg3: ['MT37'],
    xiangyaochumo: ['A4', 'A8', 'A9', 'MT62'],
    xinxin2: ['B02'],
    yishizhishen: ['MT8'],
    zd1: ['MH9', 'MT10'],
    zhenshishenghuo: ['MT52', 'MT131', 'MT132'],
    CatLegend: ['MT20', 'MT63', 'MT67'],
    cxzl_wycx: ['MT304', 'MT777'],
    evernight: ['MT30'],
    bits1: ['MT28'],
    bits2: ['dltsd5'],
    bits114514: ['bysl7', 'byxd3', 'mkpy6', 'sdhxmiddle1'],
    Black: ['MT12'],
    chaoshuang2: ['MT153'],
    EasyPreduct: ['PT40'],
    echo_of_the_dead: ['MT10', 'MT12', 'MT15'],
    Follow: ['MX4'],
    fuhao01: ['MT8', 'MT9'],
    ggcsggcs: [
        'machinetower15',
        'monastery2',
        'monastery3',
        'monastery8',
        'monastery11'
    ],
    guard: [
        'kzwm1',
        'kzwm4',
        'kzwm6',
        'kzwm8',
        'kzwm9',
        'kzwm10',
        'kzwm11',
        'kzwm12',
        'kzwm13',
        'kzwm14',
        'kzwm16',
        'kzwm19'
    ],
    hhzjmt: ['MT66', 'MT145'],
    intermediate: ['MT0'],
    jianta: ['MT17', 'MT24', 'MT23'],
    jidaoyixian: ['MT44', 'MT39', 'MT41', 'MT34', 'MT39', 'MT19'],
    jiuyueshengqishi: ['NH6', 'SR5', 'XY4'],
    luansha: ['MT354', 'MT358', 'MT364'],
    Magictower2015: ['DX16', 'DX14', 'DX21', 'DX31', 'MT4', 'MT12'],
    makai: ['MT140', 'MT164', 'MT147', 'MT239', 'MT245'],
    mikuxilie1: ['MT10', 'MT26_2'],
    mjqdt3: ['MT44'],
    Mota_Lilith: ['LLH11', 'LLH12'],
    motazhidianfeng: ['MT10'],
    MT1621: ['MT33'],
    Neko: ['MT285'],
    nyaraka: ['MT21', 'MT25', 'MT62', 'MT65'],
    qishibeige: ['QS20'],
    ruozhan: ['MT26', 'MT34', 'MT92', 'MT110', 'MT127', 'MT142', 'MT185'],
    shenzhishilian: ['MT10'],
    shilianzhidi: ['MT13'],
    thesage: ['MT14'],
    TheTravelofTheresia: ['MT15'],
    tiandijingwz: ['A15', 'A21', 'MT110'],
    tiandijing: ['MT41', 'MT45'],
    TSshadow: ['RANK1020'],
    wanning: ['MT4', 'MT8'],
    wdg2: ['MT3', 'n4'],
    WhiteLily_1521: ['B9'],
    xsdsj: ['MT40'],
    xxchuanshuo0: ['D35', 'D38', 'D39'],
    yesterdayReturn: ['MT6-1']
};

const labelConfig: IAutoLabelConfig = {
    allowedSize: [[13, 13]],
    allowUselessBranch: false,
    maxWallDensityStd: 0.25,
    minEnemyRatio: 0.02,
    maxEnemyRatio: 0.3,
    minWallRatio: 0.1,
    maxWallRatio: 0.6,
    minResourceRatio: 0.02,
    maxResourceRatio: 0.3,
    minDoorRatio: 0,
    maxDoorRatio: 0.2,
    minFishCount: 0,
    maxFishCount: 2,
    minEntryCount: 1,
    maxEntryCount: 4,
    ignoreIssues: true,
    customTowerFilter: info => {
        // if (info.color !== TowerColor.Blue && info.color !== TowerColor.Green) {
        //     return false;
        // }
        if (info.people < 1000) {
            return false;
        }
        if (info.name.startsWith('51') && info.name.length > 2) {
            return false;
        }
        if (info.name.startsWith('24') && info.name.length > 2) {
            return false;
        }
        if (ignoredTower.includes(info.name)) {
            return false;
        }

        return true;
    },
    customFloorFilter: floor => {
        if (floor.info.topo.graphs.length > 1) {
            return false;
        }
        if (floor.data.hasCannotInOut) {
            return false;
        }
        if (floor.info.topo.unreachable.size > 0) {
            return false;
        }
        if (ignoredFloor[floor.tower.name]?.includes(floor.mapId)) {
            return false;
        }
        if (floor.tower.name === 'Apeiria') {
            return Math.random() > 0.9;
        }
        return true;
    }
};

(async () => {
    const result = await autoLabelTowers(towerInfo, folders, labelConfig);
    // 转换格式并写入文件
    const dataset: GinkaDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: {}
    };
    result.forEach(tower => {
        tower.maps.forEach(floor => {
            const id = `${tower.tower.name}::${floor.mapId}`;
            const width = floor.data.map[0].length;
            const height = floor.data.map.length;
            const info = floor.info;
            const data: GinkaTrainData = {
                map: floor.data.map,
                size: [width, height],
                tag: Array(64).fill(0),
                val: [
                    info.globalDensity,
                    info.wallDensity,
                    0,
                    info.doorDensity,
                    info.enemyDensity,
                    info.resourceDensity,
                    info.gemDensity,
                    info.potionDensity,
                    info.keyDensity,
                    info.itemDensity,
                    info.entryCount,
                    info.specialDoorCount,
                    info.fishCount,
                    0,
                    0,
                    0
                ]
            };
            dataset.data[id] = data;
        });
    });
    await writeFile(output, JSON.stringify(dataset), 'utf-8');
    console.log(`结果已写入 ${output}`);
})();
