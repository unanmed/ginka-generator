import { writeFile } from 'fs-extra';
import { getAllFloors, parseTowerInfo } from './utils';
import { parseGinka } from './process/ginka';

const [output, ...list] = process.argv.slice(2);

(async () => {
    const towers = await Promise.all(
        list.map(v => parseTowerInfo(v, 'ginka-config.json'))
    );
    const floors = await getAllFloors(...towers);
    const results = parseGinka(floors);
    await writeFile(output, JSON.stringify(results, void 0), 'utf-8');
    const size = Object.keys(results.data).length;
    console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个地图`);
})();
