const numMap: Record<number, number> = {
    0: 0, // 空地
    1: 1, // 墙壁
    21: 2, // 钥匙
    27: 3, // 红宝石
    28: 4, // 蓝宝石
    31: 5, // 血瓶
    81: 6, // 门
    201: 7, // 弱怪
    202: 8, // 中怪
    203: 9, // 强怪
    87: 10, // 楼梯
    88: 10, // 楼梯
    161: 11, // 箭头
    162: 11, // 箭头
    163: 11, // 箭头
    164: 11, // 箭头
    53: 12 // 道具
};

export function convertFloor(
    map: number[][],
    [x, y, w, h]: [number, number, number, number],
    name: string,
    floorId: string
) {
    const clipped: number[][] = [];

    for (let nx = x; nx < x + w; nx++) {
        const row: number[] = [];
        for (let ny = y; ny < y + h; ny++) {
            const num = numMap[map[nx][ny]];
            if (num === void 0) {
                console.log(
                    `⚠️  魔塔 ${name} 的楼层 ${floorId} 中出现未知图块类型：${map[nx][ny]}`
                );
            }
            row.push(num ?? 0);
        }
        clipped.push(row);
    }

    return clipped;
}
