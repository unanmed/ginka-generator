import { readFile, writeFile } from 'fs-extra';
import { chooseFrom, DatasetMergable, mergeDataset } from './utils';

const [target, ...review] = process.argv.slice(2);
const n = getNum();

function getNum() {
    const last = review.at(-1);
    if (!last) return 1000;
    else {
        const n = parseInt(last);
        if (!n) return 1000;
        else {
            review.pop();
            return n;
        }
    }
}

(async () => {
    const datas = await Promise.all(
        review.map(async v => {
            const file = await readFile(v, 'utf-8');
            return JSON.parse(file) as DatasetMergable<any>;
        })
    );
    const targetFile = await readFile(target, 'utf-8');
    const targetData = JSON.parse(targetFile) as DatasetMergable<any>;
    const merged = mergeDataset(true, ...datas);
    const keys = Object.keys(merged.data);
    const toReview = chooseFrom(keys, n);
    const reviewData: DatasetMergable<any> = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: Object.fromEntries(toReview.map(v => [v, merged.data[v]]))
    };
    const reviewed = mergeDataset(false, targetData, reviewData);
    await writeFile(target, JSON.stringify(reviewed), 'utf-8');
})();
