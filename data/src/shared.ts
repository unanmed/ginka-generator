// 基本图块定义
export const emptyTiles = new Set([0]);
export const wallTiles = new Set([1]);
export const decorationTiles = new Set([2]);
export const commonDoorTiles = new Set([3, 4, 5]);
export const specialDoorTiles = new Set([6]);
export const keyTiles = new Set([7, 8, 9]);
export const redGemTiles = new Set([10, 11, 12]);
export const blueGemTiles = new Set([13, 14, 15]);
export const greenGemTiles = new Set([16, 17, 18]);
export const potionTiles = new Set([19, 20, 21, 22]);
export const itemTiles = new Set([23, 24, 25]);
export const enemyTiles = new Set([26, 27, 28]);
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
