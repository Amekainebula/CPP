#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n, k;
    cin >> n >> k;
    vector<vector<int>> e(n + 1);
    ff(i, 1, n)
    {
        int x, y;
        cin >> x >> y;
        e[x].pb(y);
        e[y].pb(x);
    }
    auto check = [&](int mid) -> bool
    {
        int need = 0;
        vector<int> q(n + 1, 0);
        auto dfs = [&](this auto &&dfs, int u, int fa) -> void
        {
            for (auto v : e[u])
            {
                if (v == fa)
                    continue;
                dfs(v, u);
                q[u] += q[v];
            }
            if (q[u] + 1 > mid)
            {
                need++;
                q[u] = 0;
            }
            else
                q[u]++;
        };
        dfs(1, -1);
        return need <= k;
    };
    int l = 0, r = n;
    while (l < r)
    {
        int mid = (l + r) / 2;
        if (check(mid))
            r = mid;
        else
            l = mid + 1;
    };
    cout << l << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}