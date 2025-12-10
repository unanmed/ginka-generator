import { readdir, readFile } from 'fs/promises';
import { parseFloorInfo, parseTowerInfo } from './info';
import { IAutoLabelConfig, IConvertedMapInfo, ITowerInfo } from './types';
import { join } from 'path';
import { Presets, SingleBar } from 'cli-progress';
import { convertTowerMap, runTowerCode } from './tower';

export interface ILabelResult {
    /** 塔信息列表 */
    readonly tower: ITowerInfo;
    /** 转换后的楼层列表 */
    readonly maps: IConvertedMapInfo[];
}

function addIssuePrefix(maxLength: number, path: string, content: string) {
    return `${path}: ${' '.repeat(maxLength - path.length)}${content}`;
}

/**
 * 自动标注塔地图
 * @param towerInfo 所有塔的信息路径，文件包括颜色、标签等
 * @param pathList 塔文件路径列表
 * @param config 自动标记配置
 */
export async function autoLabelTowers(
    towerInfo: string,
    pathList: string[],
    config: IAutoLabelConfig
) {
    const labelResult: ILabelResult[] = [];

    // 统计被不同规则过滤掉的楼层
    let ignoredFloorsSize = 0;
    let ignoredFloorsEnemy = 0;
    let ignoredFloorsWall = 0;
    let ignoredFloorsResource = 0;
    let ignoredFloorsDoor = 0;
    let ignoredFloorsFish = 0;
    let ignoredFloorsEntry = 0;
    let ignoredFloorsCustom = 0;
    let ignoredFloorsUseless = 0;
    let ignoredFloorsStd = 0;

    const towers = await parseTowerInfo(towerInfo);
    const paths: string[] = [];
    await Promise.all(
        pathList.map(async path => {
            const dir = await readdir(path);
            paths.push(...dir.map(v => join(path, v)));
        })
    );
    const issues: [string, string][] = [];
    const progress = new SingleBar({}, Presets.shades_classic);
    progress.start(paths.length, 0);
    let i = 0;
    for (const path of paths) {
        progress.update(++i);
        let project: string;
        let floors: string;
        try {
            project = await readFile(
                join(path, 'project', 'project.min.js'),
                'utf-8'
            );
            floors = await readFile(
                join(path, 'project', 'floors.min.js'),
                'utf-8'
            );
        } catch {
            issues.push([path, '读取塔信息失败']);
            continue;
        }
        const result = runTowerCode(project, floors);
        if (result.issue.length > 0) {
            issues.push(...result.issue.map<[string, string]>(v => [path, v]));
            continue;
        }

        const info = towers.get(result.data.firstData.name);
        if (!info) continue;
        const customPass = config.customTowerFilter?.(info) ?? true;
        if (!customPass) continue;
        const convertedMaps: IConvertedMapInfo[] = [];
        // 处理每个塔的每个楼层
        for (const [name, floor] of Object.entries(result.main.floors)) {
            const width = floor.map[0].length;
            const height = floor.map.length;
            // 尺寸不匹配
            const sizePass = config.allowedSize.some(
                ([w, h]) => w === width && h === height
            );
            if (!sizePass) {
                ignoredFloorsSize++;
                continue;
            }
            // 转换楼层
            const converted = convertTowerMap(result, floor);
            const floorInfo = parseFloorInfo(info, converted.map);
            const floorData: IConvertedMapInfo = {
                data: converted,
                tower: info,
                mapId: name,
                info: floorInfo
            };
            // 配置过滤楼层
            if (
                floorInfo.enemyDensity < config.minEnemyRatio ||
                floorInfo.enemyDensity > config.maxEnemyRatio
            ) {
                ignoredFloorsEnemy++;
                continue;
            }
            if (
                floorInfo.wallDensity < config.minWallRatio ||
                floorInfo.wallDensity > config.maxWallRatio
            ) {
                ignoredFloorsWall++;
                continue;
            }
            if (
                floorInfo.resourceDensity < config.minResourceRatio ||
                floorInfo.resourceDensity > config.maxResourceRatio
            ) {
                ignoredFloorsResource++;
                continue;
            }
            if (
                floorInfo.doorDensity < config.minDoorRatio ||
                floorInfo.doorDensity > config.maxDoorRatio
            ) {
                ignoredFloorsDoor++;
                continue;
            }
            if (
                floorInfo.fishCount < config.minFishCount ||
                floorInfo.fishCount > config.maxFishCount
            ) {
                ignoredFloorsFish++;
                continue;
            }
            if (
                floorInfo.entryCount < config.minEntryCount ||
                floorInfo.entryCount > config.maxEntryCount
            ) {
                ignoredFloorsEntry++;
                continue;
            }
            if (!config.allowUselessBranch && floorInfo.hasUselessBranch) {
                ignoredFloorsUseless++;
                continue;
            }
            if (floorInfo.wallDensityStd > config.maxWallDensityStd) {
                ignoredFloorsStd++;
                continue;
            }
            // 自定义过滤楼层
            const customPass = config.customFloorFilter?.(floorData) ?? true;
            if (!customPass) {
                ignoredFloorsCustom++;
                continue;
            }
            // 楼层过滤通过
            convertedMaps.push(floorData);
        }
        labelResult.push({ tower: info, maps: convertedMaps });
    }
    progress.stop();

    if (!config.ignoreIssues) {
        const maxLength = Math.max(...issues.map(v => v[0].length));
        issues.forEach(v => {
            console.log(addIssuePrefix(maxLength, v[0], v[1]));
        });
    }

    const totalFilted =
        ignoredFloorsSize +
        ignoredFloorsEnemy +
        ignoredFloorsWall +
        ignoredFloorsResource +
        ignoredFloorsDoor +
        ignoredFloorsFish +
        ignoredFloorsEntry +
        ignoredFloorsUseless +
        ignoredFloorsCustom;

    console.log(
        `已处理 ${labelResult.length} 个塔，共 ${labelResult.reduce(
            (prev, curr) => prev + curr.maps.length,
            0
        )} 层，过滤掉 ${totalFilted} 层:`
    );
    console.log(`尺寸过滤：${ignoredFloorsSize} 层`);
    console.log(`怪物过滤：${ignoredFloorsEnemy} 层`);
    console.log(`墙壁过滤：${ignoredFloorsWall} 层`);
    console.log(`资源过滤：${ignoredFloorsResource} 层`);
    console.log(`门过滤：${ignoredFloorsDoor} 层`);
    console.log(`咸鱼过滤：${ignoredFloorsFish} 层`);
    console.log(`入口过滤：${ignoredFloorsEntry} 层`);
    console.log(`无用节点过滤：${ignoredFloorsUseless} 层`);
    console.log(`标准差过滤：${ignoredFloorsStd} 层`);
    console.log(`自定义过滤：${ignoredFloorsCustom} 层`);

    return labelResult;
}
