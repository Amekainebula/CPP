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
void Murasame()
{
    int n, m;
    cin >> n >> m;
    vc<string> ans(n + 1);
    vc<vc<string>> s(m + 1, vc<string>(n + 1));
    vc<int> vis(n + 1, 0);
    ff(i, 1, n) cin >> ans[i];
    int x = -inf;
    ff(i, 1, m)
    {
        int cnt = 0;
        ff(j, 1, n)
        {
            cin >> s[i][j];
            if (s[i][j] == ans[j])
            {
                cnt++;
                vis[j] = 1;
            }
        }
        x = max(x, cnt);
    }
    ff(i, 1, n)
    {
        if (!vis[i])
        {
            cout << -1 << endl;
            return;
        }
    }
    cout << n + 2 * (n - x) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}