import { createConnection, Socket } from 'net';
import { chooseFrom, FloorData, readOne } from './utils';
import { MinamoTrainData } from './types';
import { generateTrainData } from './process/minamo';

const SOCKET_FILE = '../tmp/ginka_uds';
const [refer, replayPath = '../datasets/replay.bin'] = process.argv.slice(2);

let id = 0;

function readMap(count: number, arr: number[], h: number, w: number) {
    const area = w * h;

    const maps: number[][][] = Array.from<number[][]>({
        length: count
    }).map(() => {
        return Array.from<number[]>({ length: h }).map(() => {
            return Array.from<number>({ length: w }).fill(0);
        });
    });

    arr.forEach((v, i) => {
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

        return generateTrainData(v, id2, floor.map, map, size1, false, false, false);
    });
    return data.flat();
}

const enum ReceiverStatus {
    Header,
    Content
}

class DataReceiver {
    static active?: DataReceiver
    /** 接收状态 */
    private status: ReceiverStatus = ReceiverStatus.Header;

    private received: number[] = []
    private count: number = 0;
    private h: number = 0;
    private w: number = 0;

    receive(buf: Buffer): [number[][][], number, number, number] | null {
        // 数据通讯 node 输入协议，单位字节：
        // 2 - Tensor count; 1 - Map height; 1 - Map Width; N*1*H*W - Map tensor, int8 type.
        switch (this.status) {
            case ReceiverStatus.Header:
                this.count = buf.readInt16BE();
                this.h = buf.readInt8(2);
                this.w = buf.readInt8(3);
                this.received.push(...buf.subarray(4));
                this.status = ReceiverStatus.Content;
                break;
            case ReceiverStatus.Content:
                this.received.push(...buf);
                break
        }
        if (this.received.length === this.count * this.h * this.w) {
            delete DataReceiver.active;
            return [readMap(this.count, this.received, this.h, this.w), this.count, this.h, this.w];
        } else {
            return null;
        }
    }

    static check(buf: Buffer) {
        if (this.active) {
            return this.active.receive(buf);
        } else {
            this.active = new DataReceiver();
            return this.active.receive(buf);
        }
    }
}

(async () => {
    const referTower = await readOne(refer);
    const keys = [...referTower.keys()];

    const client = createConnection(SOCKET_FILE, () => {
        console.log(`UDS IPC connected successfully.`);
    });

    client.on('data', async buffer => {
        const data = DataReceiver.check(buffer);
        if (!data) return;

        const [map, count, h, w] = data;
        const simData = map.map(v => generateGANData(keys, referTower, v));
        const rc = 0;
        const compareData = simData.flat();

        // 数据通讯 node 输出协议，单位字节：
        // 2 - Tensor count; 2 - Replay count. Replay is right behind train data;
        // 1*tc - Compare count for every map tensor delivered.
        // 2*4*(N+rc) - Vision similarity and topo similarity, like vis, topo, vis, topo;
        // N*1*H*W - Compare map for every map tensor. rc*2*H*W - Replay map tensor.
        const toSend = Buffer.alloc(
            2 + // Tensor count
                2 + // Replay count
                1 * count + // Compare count
                2 * 4 * (compareData.length + rc) + // Similarity data
                compareData.length * 1 * h * w + // Compare map
                rc * 2 * h * w, // Replay map
            0
        );
        console.log(
            2,
            2,
            count,
            2 * 4 * (compareData.length + rc),
            compareData.length * 1 * h * w,
            rc * 2 * h * w,
            compareData.length,
            rc
        );
        
        let offset = 0;
        toSend.writeInt16BE(count); // Tensor count
        toSend.writeInt16BE(0, 2); // Replay count
        offset += 2 + 2;
        // Compare count
        toSend.set(
            simData.map(v => v.length),
            offset
        );
        offset += 1 * count;
        // Similarity data
        compareData.forEach(v => {
            // console.log(v.visionSimilarity, v.topoSimilarity);
            
            toSend.writeFloatBE(v.visionSimilarity, offset);
            offset += 4;
            toSend.writeFloatBE(v.topoSimilarity, offset);
            offset += 4;
        });
        // Compare map
        toSend.set(
            new Uint8Array(compareData.map(v => v.map1).flat(3)),
            offset // Set from Compare map
        );
        offset += compareData.length * 1 * h * w;

        client.write(toSend);
    });

    client.on('end', () => {
        console.log(`Connection lose.`);
    });

    client.on('error', () => {
        client.end();
    });
})();
