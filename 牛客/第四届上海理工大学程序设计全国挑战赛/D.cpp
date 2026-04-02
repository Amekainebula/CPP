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
void solve()
{
    int n, m;
    cin >> n >> m;
    vc<string> s(n + 1);
    ff(i, 0, n - 1) cin >> s[i];
    int R = 0, B = 0, Y = 0;
    ff(i, 0, n - 1) ff(j, 0, m - 1)
    {
        // cout << s[i][j] << " " << "\n"[j == m - 1];
        if (s[i][j] == '1')
            R++;
        else if (s[i][j] == '2')
            B++;
        else if (s[i][j] == '3')
            Y++;
    }
    vc<int> ans(4, 0);
    ans[2] = Y - B;
    ans[1] = (2 * R - Y) / 3;
    ans[3] = ans[1] - R + B;
    cout << ans[1] << " " << ans[2] << " " << ans[3] << endl;
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