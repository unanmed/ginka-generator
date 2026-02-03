// 基本图块定义
export const emptyTiles = new Set([0]);
export const wallTiles = new Set([1]);
export const decorationTiles = new Set([2]);
export const commonDoorTiles = new Set([3]);
export const specialDoorTiles = new Set([6]);
export const keyTiles = new Set([7]);
export const redGemTiles = new Set([10]);
export const blueGemTiles = new Set([13]);
export const greenGemTiles = new Set([16]);
export const potionTiles = new Set([19]);
export const itemTiles = new Set([23]);
export const enemyTiles = new Set([26]);
export const entryTiles = new Set([29]);

// 组合图块定义
export const doorTiles = commonDoorTiles.union(specialDoorTiles);
export const gemTiles = redGemTiles.union(blueGemTiles).union(greenGemTiles);
export const resourceTiles = keyTiles
    .union(gemTiles)
    .union(potionTiles)
    .union(itemTiles);
export const nonEmptyTiles = wallTiles
    .union(doorTiles)
    .union(resourceTiles)
    .union(enemyTiles)
    .union(enemyTiles);
