#include <bits/stdc++.h>
// #define int long long
const int p = 1e9 + 7;
using namespace std;
void solve()
{
    int n, q;
    cin >> n >> q;
    vector<int> ok(2000 + 5, -1);
    vector<int> a(2005, 0), qs(q + 1, 0);
    for (int i = 1; i <= n; i++)
    {
        int x;
        cin >> x;
        a[x]++;
    }
    vector<pair<int, int>> aa, vis;
    for (int i = 1; i <= 2000; i++)
    {
        if (a[i] == 0)
            continue;
        aa.push_back({i, a[i]});
    }
    int cnt = 0;
    int idx = -1;
    for (int i = 1; i <= q; i++)
    {
        cin >> qs[i];
        if (qs[i] == 1)
        {
            if (idx == -1)
            {
                idx = i;
            }
            cnt++;
        }
        if (qs[i] != -1 || i == q)
        {
            if (cnt > 1)
                vis.push_back({idx, cnt});
            idx = -1;
            cnt = 0;
        }
    }
    vector<int> l1n(q + 2, 0), mx(q + 2, 0);
    for (int i = q; i >= 1; i--)
    {
        l1n[i] = (qs[i] == 1 ? l1n[i + 1] + 1 : 0);
    }
    for (int i = q; i >= 1; i--)
    {
        mx[i] = max(mx[i + 1], qs[i]);
    }
    vector<int> sub(q + 5, 0);
    vector<int> ans(q + 1, 0);
    int sb = 0;
    for (auto [w, c] : aa)
    {
        idx = 0;
        for (int j = 1; j <= q; j++)
        {
            sb++;
            if (sb >= 1e6 * 600)
            {
                for (int j = 1; j <= q; j++)
                {
                    cout << 0 << " ";
                    return;
                }
            }
            if (w == 0)
                break;
            if (w == 1)
            {
                sub[j] += c;
                sub[j + l1n[j]] -= c;
                break;
            }
            if (mx[j] == 1)
            {
                sub[j] += c * w;
                break;
            }
            if (idx < vis.size() && j == vis[idx].first)
            {
                ans[j] += w * c * vis[idx].second;
                j += vis[idx].second - 1;
                idx++;
                continue;
            }
            w /= qs[j];
            ans[j] += w * c;
        }
    }
    int now = 0;
    for (int j = 1; j <= q; j++)
    {
        now += sub[j];
        // cerr << sub[j] << " ";
        cout << ans[j] + now << '\n';
    }
}

signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);

    int T = 1;
    // cin>>T;
    while (T--)
        solve();
    return 0;
}
/*
        int n, m;
    cin >> n >> m;
    vector<vector<int>> a(n + 1, vector<int>(n + 1, 0)), b(m + 1, vector<int>(m + 1, 0));

    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            cin >> a[i][j];
        }
    }
    for (int i = 1; i <= m; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            cin >> b[i][j];
        }
    }
    int ans = 0;
    for (int i = 1; i <= n - m + 1; ++i)
    {
        for (int j = 1; j <= n - m + 1; ++j)
        {
            for (int k = 1; k <= m; k++)
            {
                for (int l = 1; l <= m; ++l)
                {
                    int tmp = a[i + k - 1][j + l - 1] ^ b[k][l];
                    // cerr << i + k - 1 << " " << j + l - 1 << " " << k << " " << l << " " << tmp << '\n';
                    ans = (ans + tmp) % p;
                }
            }
        }
    }
    cout << ans << '\n';
    */