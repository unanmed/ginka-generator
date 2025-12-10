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
    'xieyuhouxu'
];

const ignoredFloor: Record<string, string[]> = {
    cxzl2: ['MN5', 'MN7', 'MN8', 'MN49', 'MN57', 'MQ77', 'MT85', 'MZ75'],
    cxzl: ['MT63', 'MT186', 'MT256', 'MT304', 'MT313', 'MT314', 'MT334'],
    gl: ['MH2'],
    gooTower2: ['MT96'],
    levelstower: ['LV0c'],
    lzlltete: ['MT0'],
    myslmdrydtwo: ['MT49'],
    OneStepToHell: ['MT02', 'MT05', 'MT07', 'MT09', 'MT44'],
    ousha: ['MT152', 'MT158', 'MT160'],
    Pacificwar: ['MT102', 'MT131'],
    tiandijinghz: ['MT54', 'MT41', 'A21', 'MT50', 'MT110'],
    wdg3: ['MT37'],
    xiangyaochumo: ['A4', 'A8', 'A9', 'MT62'],
    xinxin2: ['B02'],
    yishizhishen: ['MT8'],
    zd1: ['MH9', 'MT10'],
    zhenshishenghuo: ['MT52', 'MT131', 'MT132']
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
        if (info.color !== TowerColor.Blue && info.color !== TowerColor.Green) {
            return false;
        }
        if (info.people < 1000) {
            return false;
        }
        if (info.name.startsWith('51') && info.name.length > 2) {
            return Math.random() > 0.98;
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
        if (floor.info.topo.unreachable.size > 5) {
            return false;
        }
        if (ignoredFloor[floor.tower.name]?.includes(floor.mapId)) {
            return false;
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
