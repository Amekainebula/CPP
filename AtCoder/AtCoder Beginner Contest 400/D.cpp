#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
int next1[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
void solve()
{
    int n, m;
    cin >> n >> m;
    vc<vc<char>> mp(n + 1, vc<char>(m + 1));
    vc<vc<int>> dis(n + 1, vc<int>(m + 1, -1));
    ff(i, 1, n) ff(j, 1, m) cin >> mp[i][j];
    int x, y, b, e;
    cin >> x >> y >> b >> e;
    auto bfs = [&](int x, int y)
    {
        deque<array<int, 3>> q;
        q.push_back({x, y, 0});
        while (!q.empty())
        {
            auto [u, v, c] = q.front();
            q.pop_front();
            if (dis[u][v] != -1)
                continue;
            dis[u][v] = c;
            //cout << u << ' ' << v << ' ' << c << endl;
            ff(k, 0, 3) ff(t, 1, 2)
            {
                int tx = u + next1[k][0] * t, ty = v + next1[k][1] * t;
                if (tx < 1 || tx > n || ty < 1 || ty > m)
                    continue;
                if (mp[tx][ty] == '.' && t == 1)
                {
                    q.push_front({tx, ty, c});
                }
                else
                {
                    q.push_back({tx, ty, c + 1});
                }
            }
        }
    };
    bfs(x, y);
    //ff(i, 1, n) ff(j, 1, m) cout << dis[i][j] << " \n"[j == m];
    cout << dis[b][e] << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}