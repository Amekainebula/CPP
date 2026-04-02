// checker.cpp
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int T = inf.readInt();

    for (int tc = 1; tc <= T; tc++) {
        long long expected = ans.readLong();
        long long participant = ouf.readLong();

        if (expected != participant) {
            quitf(_wa,
                  "Wrong answer on test %d: expected %lld, found %lld",
                  tc, expected, participant);
        }
    }

    if (!ouf.seekEof()) {
        quitf(_pe, "Extra output");
    }

    quitf(_ok, "All tests passed.");
}
