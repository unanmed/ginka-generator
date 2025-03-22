import { readFile, writeFile } from 'fs-extra';
import { DatasetMergable, mergeDataset } from './utils';

const [output, ...datasets] = process.argv.slice(2);

(async () => {
    const data = await Promise.all(
        datasets.map(async v => {
            const file = await readFile(v, 'utf-8');
            return JSON.parse(file) as DatasetMergable<any>;
        })
    );
    const merged = mergeDataset(...data);
    await writeFile(output, JSON.stringify(merged), 'utf-8');
})();
