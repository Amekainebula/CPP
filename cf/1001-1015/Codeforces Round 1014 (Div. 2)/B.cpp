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
    int n;
    string a, b;
    cin >> n >> a >> b;
    int c11 = 0, c12 = 0, c21 = 0, c22 = 0;
    for (int i = 0; i < n; i++)
    {
        if (a[i] == '1')
        {
            if (i % 2 == 0)
                c11++;
            else
                c12++;
        }
        if (b[i] == '1')
        {
            if (i % 2 == 0)
                c21++;
            else
                c22++;
        }
    }
    if (c11 <= n / 2 - c22 && c12 <= n - n / 2 - c21 || c21 <= n - n / 2 - c12 && c22 <= n / 2 - c11)
    {
        cout << "YES\n";
    }
    else
    {
        cout << "NO\n";
    }
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
        solve();
    }
    return 0;
}