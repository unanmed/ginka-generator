import { readFile, writeFile } from 'fs-extra';
import { GinkaDataset } from './types';
import { chooseFrom } from './utils';

const [outputTrain, outputEval, input, ratioStr] = process.argv.slice(2);
const ratio = parseFloat(ratioStr);

(async () => {
    const data = await readFile(input, 'utf-8');
    const dataJSON = JSON.parse(data) as GinkaDataset;
    const keys = Object.keys(dataJSON.data);
    const length = keys.length;
    const toEval = chooseFrom(keys, Math.floor(length * ratio));
    const toTrain = [...new Set(keys).difference(new Set(toEval))];
    const trainData: GinkaDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: {}
    };
    toTrain.forEach(v => {
        trainData.data[v] = dataJSON.data[v];
    });
    const evalData: GinkaDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: {}
    };
    toEval.forEach(v => {
        evalData.data[v] = dataJSON.data[v];
    });
    await writeFile(outputTrain, JSON.stringify(trainData), 'utf-8');
    await writeFile(outputEval, JSON.stringify(evalData), 'utf-8');
})();
