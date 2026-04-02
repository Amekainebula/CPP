#include <bits/stdc++.h>
#define ll long long
#define ull unsigned long long
#define i128 __int128
#define d64 long double
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
int net[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
char netc[4] = {'u', 'l', 'r', 'd'};
void solve()
{
    int n, m, k;
    cin >> n >> m >> k;
    vector<vector<int>> a(n + 1, vector<int>(m + 1));
    vector<vector<char>> ans(n + 1, vector<char>(m + 1));
    vector<pair<int, char>> sn(n * m + 100);
    ff(i, 1, n) ff(j, 1, m)
    {
        int x;
        cin >> x;
        a[i][j] = x;
        if (x == 1)
            ans[i][j] = 'O';
        else
            ans[i][j] = 'X';
    }
    ff(i, 1, k)
    {
        int x, y;
        char c;
        cin >> x >> y >> c;
        if (ans[x][y] != 'O')
            continue;
        // cout << x << " " << y << " " << c << " " << ans[x][y] << endl;
        if (c == 'U' || c == 'D' || c == 'L' || c == 'R')
            ans[x][y] = c;
        else
        {
            sn[(x - 1) * m + y].fi = i;
            sn[(x - 1) * m + y].se = c;
            ans[x][y] = c;
            bool ok = 0;
            auto solve1 = [&](int x, int y, int tx, int ty)
            {
                int temp = (tx - 1) * m + ty;
                if (temp <= 0 || temp > n * m)
                    return;
                if (sn[temp].fi != 0)
                {
                    // cout<<tx<<" "<<ty<<endl;
                    ans[tx][ty] = toupper(ans[tx][ty]);
                    ans[x][y] = 'O';
                    sn[temp].fi = 0;
                    sn[(x - 1) * m + y].fi = 0;
                    ok = 1;
                }
            };
            if (c == 'l')
            {
                int tx = x, ty = y - 1;
                solve1(x, y, tx, ty);
            }
            else if (c == 'u')
            {
                int tx = x - 1, ty = y;
                solve1(x, y, tx, ty);
            }
            else if (c == 'r')
            {
                int tx = x, ty = y + 1;
                solve1(x, y, tx, ty);
            }
            else if (c == 'd')
            {
                int tx = x + 1, ty = y;
                solve1(x, y, tx, ty);
            }
            if (!ok)
            {
                vector<array<int, 3>> v;
                ff(i, 0, 4)
                {
                    int tx = x + net[i][0], ty = y + net[i][1];
                    int temp = (tx - 1) * m + ty;
                    if (temp <= 0 || temp > n * m)
                        continue;
                    if (sn[temp].fi != 0 && sn[temp].se == netc[i])
                        v.pb({sn[temp].fi, tx, ty});
                }
                auto cmp = [&](array<int, 3> a, array<int, 3> b)
                {
                    auto [_, x1, y1] = a;
                    auto [__, x2, y2] = b;
                    return _ > __;
                };
                sort(all(v), cmp);
                if (v.size() > 0)
                {
                    auto [_, tx, ty] = v[0];
                    // cout<<tx<<" "<<ty<<endl;
                    ans[x][y] = toupper(ans[x][y]);
                    ans[tx][ty] = 'O';
                    sn[(x - 1) * m + y].fi = 0;
                    sn[(tx - 1) * m + ty].fi = 0;
                }
            }
        }
        // cout << x << " " << y << " " << c << " " << ans[x][y] << endl;
    }
    ff(i, 1, n)
    {
        ff(j, 1, m)
        {
            cout << ans[i][j];
        }
        cout << endl;
    }
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