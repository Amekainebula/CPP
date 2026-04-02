#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
int nex[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
char dis[4] = {'^', '<', 'v', '>'};
void Murasame()
{
    queue<pii> q;
    int n, m;
    cin >> n >> m;
    vc<vc<char>> maps(n + 1, vc<char>(m + 1));
    ff(i, 1, n) ff(j, 1, m)
    {
        cin >> maps[i][j];
        if (maps[i][j] == 'E')
        {
            q.push({i, j});
        }
    }
    while (!q.empty())
    {
        int x = q.front().fi, y = q.front().se;
        q.pop();
        ff(i, 0, 3)
        {
            int nx = x + nex[i][0], ny = y + nex[i][1];
            if (nx > 0 && nx <= n && ny > 0 && ny <= m && maps[nx][ny] == '.')
            {
                maps[nx][ny] = dis[i];
                q.push({nx, ny});
            }
        }
    }
    ff(i, 1, n)
    {
        ff(j, 1, m)
        {
            cout << maps[i][j];
        }
        cout << endl;
    }
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
        Murasame();
    }
    return 0;
}