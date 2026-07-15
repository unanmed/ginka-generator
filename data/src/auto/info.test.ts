import { describe, expect, it } from 'vitest';
import { parseFloorInfo } from './info';
import {
    IAutoLabelConfig,
    IMapTileConverter,
    ITowerInfo,
    ResourceType,
    TowerColor
} from './types';

class MockTileConverter implements IMapTileConverter {
    getLabeledTile(tile: number): number {
        return tile;
    }

    isEmpty(tile: number): boolean {
        return tile === 0;
    }

    isEntry(tile: number): boolean {
        return tile === 5;
    }

    isDoor(tile: number): boolean {
        return tile === 2 || tile === 6;
    }

    isEnemy(tile: number): boolean {
        return tile === 4;
    }

    isResource(tile: number): boolean {
        return tile === 3;
    }

    getNoPass(tile: number): boolean {
        return tile === 1;
    }

    getCannotIn(): number {
        return 0;
    }

    getCannotOut(): number {
        return 0;
    }

    getResource(tile: number): Map<ResourceType, number> {
        if (tile === 3) {
            return new Map([[ResourceType.Item, 1]]);
        }
        return new Map();
    }
}

const tower: ITowerInfo = {
    authorId: 1,
    color: TowerColor.Blue,
    comment: 0,
    competition: false,
    floors: 1,
    id: 1,
    name: 'test-tower',
    people: 1000,
    tag: [],
    title: 'test tower',
    topuser: [],
    win: 0,
    designrate: [],
    hardrate: []
};

const config: IAutoLabelConfig = {
    classes: {
        empty: 0,
        wall: 1,
        decoration: 16,
        commonDoors: [2],
        specialDoors: [6, 6],
        keys: [3],
        redGems: [3],
        blueGems: [3],
        greenGems: [3],
        potions: [3],
        items: [3],
        enemies: [4],
        entry: 5
    },
    allowedSize: [[5, 5]],
    allowLargeDoorCluster: false,
    allowLargeEnemyCluster: false,
    allowIdleBranch: false,
    allowRepeatedGuardIdleBranch: false,
    allowUselessBranch: false,
    minEnemyRatio: 0,
    maxEnemyRatio: 1,
    minWallRatio: 0,
    maxWallRatio: 1,
    minResourceRatio: 0,
    maxResourceRatio: 1,
    minDoorRatio: 0,
    maxDoorRatio: 1,
    minFishCount: 0,
    maxFishCount: 10,
    minEntryCount: 0,
    maxEntryCount: 10,
    maxWallDensityStd: Infinity,
    maxEmptyArea: Infinity,
    maxResourceArea: Infinity,
    heatmapKernel: 1,
    guassainRadius: 0,
    ignoreIssues: true
};

function parseTestFloor(map: number[][]) {
    return parseFloorInfo(
        tower,
        map,
        map,
        [],
        config,
        new MockTileConverter(),
        'F1'
    );
}

function parseTestFloorWithConverter(
    map: number[][],
    converter: IMapTileConverter
) {
    return parseFloorInfo(tower, map, map, [], config, converter, 'F1');
}

class BlockedDirectionConverter extends MockTileConverter {
    getCannotIn(): number {
        return 0b1111;
    }

    getCannotOut(): number {
        return 0b1111;
    }
}

describe('parseFloorInfo useless branch detection', () => {
    it('marks a branch with only one grid-level passable direction as useless', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 5, 2, 1, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.hasUselessBranch).toBe(true);
    });

    it('marks a branch as useless when every backside candidate loses entry reachability and has no resource', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 5, 2, 0, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.hasUselessBranch).toBe(true);
    });

    it('keeps a branch when its disconnected backside can still reach resource through other branches', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1, 1],
            [1, 5, 2, 4, 3, 1],
            [1, 1, 1, 1, 1, 1]
        ]);

        expect(floor.hasUselessBranch).toBe(false);
    });
});

describe('parseFloorInfo continuous branch cluster detection', () => {
    it('marks a door cluster larger than 3 using same-type BFS connectivity', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 5, 2, 2, 1],
            [1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.maxDoorClusterSize).toBe(4);
        expect(floor.hasLargeDoorCluster).toBe(true);
        expect(floor.hasLargeEnemyCluster).toBe(false);
    });

    it('keeps a same-type branch cluster whose size is exactly 3', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 5, 4, 4, 1],
            [1, 1, 1, 4, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.maxEnemyClusterSize).toBe(3);
        expect(floor.hasLargeEnemyCluster).toBe(false);
        expect(floor.hasLargeDoorCluster).toBe(false);
    });

    it('does not merge mixed door-enemy adjacency into one same-type cluster', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 5, 4, 4, 2, 4, 4],
            [1, 1, 1, 1, 1, 1, 1]
        ]);

        expect(floor.maxEnemyClusterSize).toBe(2);
        expect(floor.maxDoorClusterSize).toBe(1);
        expect(floor.hasLargeEnemyCluster).toBe(false);
        expect(floor.hasLargeDoorCluster).toBe(false);
    });
});

describe('parseFloorInfo idle branch detection', () => {
    it('marks a door branch with exactly one topology neighbor as idle', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 0, 2, 1, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.idleDoorBranchCount).toBe(1);
        expect(floor.idleEnemyBranchCount).toBe(0);
        expect(floor.hasIdleBranch).toBe(true);
    });

    it('marks an enemy branch with exactly one topology neighbor as idle', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 0, 4, 1, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.idleDoorBranchCount).toBe(0);
        expect(floor.idleEnemyBranchCount).toBe(1);
        expect(floor.hasIdleBranch).toBe(true);
    });

    it('does not mark a branch idle when passing it exposes multiple topology neighbors', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 0, 4, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.idleDoorBranchCount).toBe(0);
        expect(floor.idleEnemyBranchCount).toBe(0);
        expect(floor.hasIdleBranch).toBe(false);
    });
});

describe('parseFloorInfo repeated guard idle detection', () => {
    it('marks same-type branches that repeatedly guard the same merged regions', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 3, 4, 0, 1],
            [1, 3, 3, 4, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.repeatedGuardDoorBranchCount).toBe(0);
        expect(floor.repeatedGuardEnemyBranchCount).toBe(2);
        expect(floor.hasRepeatedGuardIdleBranch).toBe(true);
    });

    it('does not merge mixed door-enemy guards into one repeated guard pattern', () => {
        const floor = parseTestFloor([
            [1, 1, 1, 1, 1],
            [1, 3, 2, 0, 1],
            [1, 3, 3, 4, 1],
            [1, 1, 1, 1, 1]
        ]);

        expect(floor.repeatedGuardDoorBranchCount).toBe(0);
        expect(floor.repeatedGuardEnemyBranchCount).toBe(0);
        expect(floor.hasRepeatedGuardIdleBranch).toBe(false);
    });
});

describe('parseFloorInfo wall-only passability detection', () => {
    it('ignores cannotIn and cannotOut flags in the useless branch quick check', () => {
        const floor = parseTestFloorWithConverter(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 5, 2, 4, 3, 1],
                [1, 1, 1, 1, 1, 1]
            ],
            new BlockedDirectionConverter()
        );

        expect(floor.hasUselessBranch).toBe(false);
    });
});
