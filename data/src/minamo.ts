import { writeFile } from 'fs-extra';
import { readOne, getAllFloors, parseTowerInfo } from './utils';
import { generateAssignedData, parseMinamo } from './process/minamo';

const [output, ...list] = process.argv.slice(2);
// 判断 assigned 模式，此模式下只会对前两个塔处理，会在这两个塔之间对比，而单个塔的地图不会对比
const assigned = list.at(-1)?.startsWith('assigned');
const assignedCount = parseAssigned(list.at(-1)!);
if (assigned) list.pop();

function parseAssigned(arg: string): [number, number] {
    const p = arg.slice(9);
    const [a, b] = p.split(':');
    return [parseInt(a) || 100, parseInt(b) || 100];
}

(async () => {
    if (!assigned) {
        const towers = await Promise.all(
            list.map(v => parseTowerInfo(v, 'minamo-config.json'))
        );
        const floors = await getAllFloors(...towers);
        const results = parseMinamo(floors);
        await writeFile(output, JSON.stringify(results, void 0), 'utf-8');
        const size = Object.keys(results.data).length;
        console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个组合`);
    } else {
        const [tower1, tower2] = list;
        if (!tower1 || !tower2) {
            console.log(`⚠️  assigned 模式下必须传入两个塔！`);
            return;
        }
        const data1 = await readOne(tower1);
        const data2 = await readOne(tower2);
        const results = generateAssignedData(data1, data2, assignedCount);
        await writeFile(output, JSON.stringify(results, void 0), 'utf-8');
        const size = Object.keys(results.data).length;
        console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个组合`);
    }
})();
