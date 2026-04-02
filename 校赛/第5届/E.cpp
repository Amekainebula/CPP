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
int net[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
void solve()
{
    int n, m;
    cin >> n >> m;
    vector<vector<char>> map1(n + 1, vector<char>(m + 1)),
        map2(n + 1, vector<char>(2 * m + 1)),
        map3(2 * n + 1, vector<char>(2 * m + 1));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            cin >> map1[i][j];
    auto bfs = [&](vector<vector<char>> &map, int n, int m)
    {
        // cout<<endl;
        queue<array<int, 3>> q;
        q.push({1, 1, 0});
        vector<vector<bool>> vis(n + 1, vector<bool>(m + 1, false));
        vis[1][1] = true;
        int cnt = 0;
        while (!q.empty())
        {
            auto [u, v, cnt] = q.front();
            // cout << u << " " << v << " " << cnt << endl;
            if (map[u][v] == '*')
            {
                if (u == n && v == m || 
                    u == n / 2 && v == m / 2 || 
                    u == n / 2 && v == m)
                    return cnt;
            }
            cnt++;
            q.pop();
            for (int i = 0; i < 4; i++)
            {
                int x = u + net[i][0], y = v + net[i][1];
                if (x < 1 || x > n || y < 1 || y > m)
                    continue;
                if (!vis[x][y] && map[x][y] == '*')
                {
                    vis[x][y] = true;
                    q.push({x, y, cnt});
                    if (x == n && y == m ||
                        x == n / 2 && y == m / 2 ||
                        x == n / 2 && y == m)
                        return cnt;
                }
            }
        }
        return -1LL;
    };
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            map3[i][j] = map1[i][j];
            map3[2 * n - i + 1][j] = map1[i][j];
            map3[i][2 * m - j + 1] = map1[i][j];
            map3[2 * n - i + 1][2 * m - j + 1] = map1[i][j];
        }
    }
    int cnt3 = bfs(map3, 2 * n, 2 * m);
    cout << cnt3 << endl;
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