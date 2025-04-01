import { createConnection, Socket } from 'net';
import { chooseFrom, FloorData, readOne } from './utils';
import { MinamoTrainData } from './types';
import { generateTrainData } from './process/minamo';

const SOCKET_FILE = '../tmp/ginka_uds';
const [refer] = process.argv.slice(2);

let id = 0;

function readMap(count: number, buffer: Buffer, h: number, w: number) {
    const area = w * h;

    const maps: number[][][] = Array.from<number[][]>({
        length: count
    }).map(() => {
        return Array.from<number[]>({ length: h }).map(() => {
            return Array.from<number>({ length: w }).fill(0);
        });
    });

    buffer.subarray(4).forEach((v, i) => {
        const n = Math.floor(i / area);
        const y = Math.floor((i % area) / w);
        const x = i % w;
        maps[n][y][x] = v;
    });

    return maps;
}

function generateGANData(
    keys: string[],
    refer: Map<string, FloorData>,
    map: number[][]
) {
    const id2 = `$${id++}`;
    const toTrain = chooseFrom(keys, 4);
    const data = toTrain.map<MinamoTrainData[]>(v => {
        const floor = refer.get(v);
        if (!floor) return [];
        const size1: [number, number] = [floor.map[0].length, floor.map.length];
        const size2: [number, number] = [map[0].length, map.length];
        if (size1[0] !== size2[0] || size1[1] !== size2[1]) return [];

        return generateTrainData(v, id2, floor.map, map, size1);
    });
    return data.flat();
}

(async () => {
    const referTower = await readOne(refer);
    const keys = [...referTower.keys()];

    const client = createConnection(SOCKET_FILE, () => {
        console.log(`UDS IPC connected successfully.`);
        // 发送四字节数据表示连接成功
        client.write(new Uint8Array([0x00, 0x00, 0x00, 0x00]));
    });

    client.on('data', buffer => {
        // 暂时不考虑流式传输，如果后续数据量非常大，再考虑优化
        // 数据通讯 node 输入协议，单位字节：
        // 2 - Tensor count; 1 - Map height; 1 - Map Width; N*1*H*W - Map tensor, int8 type.
        const count = buffer.readInt16BE();
        if (buffer.length - 4 !== count * 32 * 32) {
            client.write(`ERROR: byte length not match.`);
            return [];
        }
        const h = buffer.readInt8(2);
        const w = buffer.readInt8(3);
        const map = readMap(count, buffer, h, w);
        const simData = map.map(v => generateGANData(keys, referTower, v));
        const rc = 0;
        const compareData = simData.flat();
        const reviewData: MinamoTrainData[] = [];

        // 数据通讯 node 输出协议，单位字节：
        // 2 - Tensor count; 2 - Review count. Review is right behind train data;
        // 1*tc - Compare count for every map tensor delivered.
        // 2*4*(N+rc) - Vision similarity and topo similarity, like vis, topo, vis, topo;
        // N*1*H*W - Compare map for every map tensor. rc*2*H*W - Review map tensor.
        const toSend = Buffer.alloc(
            2 + // Tensor count
                2 + // Review count
                count + // Compare count
                2 * (count + rc) + // Similarity data
                compareData.length * 1 * h * w + // Compare map
                rc * 2 * h * w, // Review map
            0
        );
        let offset = 0;
        toSend.writeInt16BE(count); // Tensor count
        toSend.writeInt16BE(0, 2); // Review count
        offset += 2 + 2;
        // Compare count
        toSend.set(
            simData.map(v => v.length),
            offset
        );
        offset += count;
        // Similarity data
        compareData.forEach(v => {
            toSend.writeFloatBE(v.visionSimilarity, offset);
            offset += 4;
            toSend.writeFloatBE(v.topoSimilarity, offset);
            offset += 4;
        });
        // Compare map
        toSend.set(
            compareData.map(v => v.map1).flat(2),
            offset // Set from Compare map
        );
        offset += compareData.length * 1 * h * w;
        // Review map
        toSend.set(
            reviewData.map(v => [v.map1, v.map2]).flat(3),
            offset // Set from last chunk
        );

        client.write(toSend);
    });

    client.on('end', () => {
        console.log(`Connection lose.`);
    });

    client.on('error', () => {
        client.end();
    });
})();
