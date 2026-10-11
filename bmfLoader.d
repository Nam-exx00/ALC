module bmfLoader;

/*
    BMP Loader

    Copyright 2026 Nam_exx00

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

import std.conv : to;
import ALC;

string[] usd;
string[] asd;

private ulong readLE(ubyte[] data, ulong pos, ulong n) {
    ulong v = 0;
    foreach (i; 0 .. n)
        v |= cast(ulong)data[pos + i] << (8 * i);
    return v;
}

void loadBMF(ubyte[] bmf) {
    if (bmf.length < 11)
        throw new Exception("Less length of Block based Model knowledge Format File.");

    if (bmf[0] != 0x42 || bmf[1] != 0x4D || bmf[2] != 0x46)
        throw new Exception("Invalid Block based Model knowledge Format Header.");

    ulong count = readLE(bmf, 3, 8);
    ulong pos = 11;

    foreach (_; 0 .. count) {
        if (pos + 9 > bmf.length)
            throw new Exception("Less length of Block based Model knowledge Format File.");

        ubyte type = bmf[pos];
        ulong len = readLE(bmf, pos + 1, 8);
        pos += 9;

        if (pos + len > bmf.length)
            throw new Exception("Less length of Block based Model knowledge Format File.");

        string content = cast(string)bmf[pos .. pos + len];
        pos += len;

        if (type == 0)
            usd ~= content;
        else asd ~= content;
    }
}