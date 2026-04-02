#include <bits/stdc++.h>
#define int long long
using namespace std;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

int randll(int l, int r) { return uniform_int_distribution<int>(l, r)(rng); }

signed main() {
    const int file_count = 20;

    filesystem::create_directories("Data");

    for (int i = 1; i <= file_count; ++i) {
        string in_path = "Data/" + to_string(i) + ".in";
        string out_path = "Data/" + to_string(i) + ".out";
        ofstream fin(in_path);
        ofstream fout(out_path);

        int n, m;
        if(i <= 10) {
            n = randll(2, 1000);
            m = randll(1, min(1000ll, n * (n - 1) / 2));
        }else {
            n = randll(2, 200000);
            m = randll(1, min(200000ll, n * (n - 1) / 2));
        }
        fin << n << ' ' << m << '\n';

        unordered_set<int> used;
        used.reserve(m * 2);

        if(i % 5 == 0 && n >= 5) {
            int cnt = randll(3, max(n / 2, 5ll));
            vector<int> tmp;
            for(int k = 0; k < cnt; k++) {
                tmp.emplace_back(randll(1, n));
            }
            for(int k = 0; k < cnt; k++) {
                int u = tmp[k], v = tmp[(k + 1) % cnt];
                int key1 = 1ll * u * n + v;
                int key2 = 1ll * v * n + u;
                used.insert(key1);
                used.insert(key2);
                fin << 1 << ' ' << u << ' ' << v << '\n';
            }
        }

        while ((int)used.size() < m * 2) {
            int u = randll(1, n);
            int v = randll(1, n);
            if (u == v) continue;

            int key1 = 1ll * u * n + v;
            int key2 = 1ll * v * n + u;
            if (used.insert(key1).second && used.insert(key2).second) {
                if (randll(1, 10) <= 3)
                    fin << 0 << " " << u << " " << v << '\n';
                else
                    fin << 1 << " " << u << " " << v << '\n';
            }
        }
    }
}
