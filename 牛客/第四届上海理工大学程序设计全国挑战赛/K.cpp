#include <bits/stdc++.h>
#define ll long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, ll>
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
int nextt[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
void solve()
{
    int n, m;
    cin >> n >> m;
    vector<vector<int>> ans(n + 1, vector<int>(m + 1, 0));
    int cnt = 0;
    int x = 1, y = 1;
    ans[1][1] = 1;
    for (int i = 2; i <= n * m;)
    {
        cnt %= 4;
        int tx = x + nextt[cnt][0], ty = y + nextt[cnt][1];
        if (tx > n || tx < 1 || ty > m || ty < 1 || ans[tx][ty])
        {
            cnt++;
            continue;
        }
        ans[tx][ty] = i;
        x = tx, y = ty;
        //cout << x << " " << y << endl;
        i++;
    }
    ff(i, 1, n) ff(j, 1, m) cout << ans[i][j] << " \n"[j == m];
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