#include <bits/stdc++.h>
// Finish Time: 2025/3/9 20:15:54 AC
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
int dx[4] = {0, -1, 0, 1};
int dy[4] = {1, 0, -1, 0};
void solve()
{
    int n, m;
    cin >> n >> m;
    vector<int> a(n * m + 1), b(n * m + 1);
    auto Encode = [&](int x, int y)
    { return (x - 1) * m + y; };
    auto Decode = [&](int id)
    { return make_pair((id - 1) / m + 1, (id - 1) % m + 1); };
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cin >> a[Encode(i, j)];
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            cin >> b[Encode(i, j)];
    bool vis[n * m + 1][4];
    int dis[n * m + 1][4];
    auto dij = [&]()
    {
        int st = Encode(1, 1);
        int td = Encode(n, m);
        for (int i = 1; i <= n * m; i++)
            for (int j = 0; j < 4; j++)
            {
                dis[i][j] = INF;
                vis[i][j] = 0;
            }
        dis[st][0] = a[st];
        priority_queue<array<int, 3>, vector<array<int, 3>>, greater<> >pq;
        pq.push({0, st, 0});
        while (!pq.empty())
        {
            auto [val, p, u] = pq.top();
            pq.pop();
            if (vis[p][u])
                continue;
            vis[p][u] = 1;
            for (int v = 0; v < 4; v++)
            {   
                if (dis[p][u] + b[p] >= dis[p][v])
                    continue;
                dis[p][v] = dis[p][u] + b[p];
                pq.push({dis[p][v], p, v});
            }
            auto [x, y] = Decode(p);
            x += dx[u];
            y += dy[u];
            int q = Encode(x, y);
            auto OutMap = [&](int x, int y)
            {
                return x < 1 || x > n || y < 1 || y > m;
            };
            if (!OutMap(x, y) && dis[p][u] + a[q] < dis[q][u])
            {
                dis[q][u] = dis[p][u] + a[q];
                pq.push({dis[q][u], q, u});
            }
        }
        return dis[td][3];
    };
    int ans = dij();
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}