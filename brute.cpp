#include <bits/stdc++.h>
using namespace std;

using i64 = long long;

void solve()
{
    int n, m, k, b;
    cin >> n >> m >> k >> b;

    vector<i64> val(n);
    vector<int> c(n);
    for (int i = 0; i < n; i++)
        cin >> val[i];
    for (int i = 0; i < n; i++)
        cin >> c[i];

    vector<tuple<int, int, i64>> edges;
    for (int i = 0; i < m; i++)
    {
        int u, v;
        i64 w;
        cin >> u >> v >> w;
        --u, --v;
        edges.push_back({u, v, w});
    }

    i64 ans = -(1LL << 60);

    for (int s = 0; s < (1 << n); s++)
    {
        int cnt = __builtin_popcount((unsigned)s);
        if (cnt < k)
            continue;

        int cnt1 = 0;
        i64 sum = 0;

        for (int i = 0; i < n; i++)
        {
            if ((s >> i) & 1)
            {
                sum += val[i];
                cnt1 += c[i];
            }
        }

        int cnt0 = cnt - cnt1;
        if (cnt0 < b || cnt1 < b)
            continue;

        for (auto [u, v, w] : edges)
        {
            if (((s >> u) & 1) && ((s >> v) & 1))
            {
                sum += w;
            }
        }

        ans = max(ans, sum);
    }

    cout << ans << '\n';
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}
