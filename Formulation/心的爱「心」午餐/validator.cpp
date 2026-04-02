// validator.cpp
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> fa;
    DSU(int n) : fa(n + 1) {
        iota(fa.begin(), fa.end(), 0);
    }
    int find(int x) {
        return fa[x] == x ? x : fa[x] = find(fa[x]);
    }
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        fa[x] = y;
        return true;
    }
};

int main(int argc, char* argv[]) {
    registerValidation(argc, argv);

    int T = inf.readInt(1, 100, "T");
    inf.readEoln();

    int sum_n = 0;
    int sum_k = 0;

    for (int tc = 1; tc <= T; tc++) {
        int n = inf.readInt(1, 200, "n");
        inf.readSpace();
        int m = inf.readInt(0, n - 1, "m");
        inf.readSpace();
        int k = inf.readInt(1, n, "k");
        inf.readSpace();
        int b = inf.readInt(0, k / 2, "b");
        inf.readEoln();

        sum_n += n;
        sum_k += k;

        // w_i
        vector<long long> w(n + 1);
        for (int i = 1; i <= n; i++) {
            w[i] = inf.readLong(-1000000000LL, 0LL, "w_i");
            if (i < n) inf.readSpace();
        }
        inf.readEoln();

        // c_i
        vector<int> c(n + 1);
        int sum_c = 0;
        for (int i = 1; i <= n; i++) {
            c[i] = inf.readInt(0, 1, "c_i");
            sum_c += c[i];
            if (i < n) inf.readSpace();
        }
        inf.readEoln();

        // 颜色数量约束
        ensuref(sum_c >= b, "sum(c_i) < b");
        ensuref(sum_c <= n - b, "sum(c_i) > n - b");

        DSU dsu(n);
        set<pair<int,int>> used;

        for (int i = 1; i <= m; i++) {
            int u = inf.readInt(1, n, "u");
            inf.readSpace();
            int v = inf.readInt(1, n, "v");
            inf.readSpace();
            long long W = inf.readLong(-1000000000LL, 0LL, "W");
            inf.readEoln();

            ensuref(u != v, "self-loop detected");

            if (u > v) swap(u, v);
            ensuref(!used.count({u, v}), "duplicate edge");
            used.insert({u, v});

            // 检查无环（森林）
            ensuref(dsu.merge(u, v), "cycle detected");
        }
    }

    ensuref(sum_n <= 200, "sum of n exceeds 200");
    ensuref(sum_k <= 200, "sum of k exceeds 200");

    inf.readEof();
}
