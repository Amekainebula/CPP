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
#define vvi vector<vi>
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
void Murasame()
{
    int n, m;
    cin >> n >> m;
    int ans = 0;
    vc<vc<int>> a(n + 5, vc<int>(m + 5));
    vc<vc<int>> prey(m + 1, vc<int>(n + 1));
    ff(i, 1, n)
    {
        int maxx = -1;
        ff(j, 1, m)
        {
            cin >> a[i][j];
            maxx = max(maxx, a[i][j]);
        }
        a[i][0] = maxx;
        ans += maxx;
    }
    ff(i, 1, m)
    {
        ff(j, 1, n)
        {
            prey[i][j] = a[j][i] - a[j][0];
            // cout << prey[i][j] << " \n"[j == n];
        }
    }
    // ff(i, 1, n) ff(j, 0, m) cout << a[i][j] << " \n"[j == m];
    // ff(i, 1, m) ff(j, 1, n) cout << prey[i][j] << " \n"[j == n];
    ff(i, 1, m) sort(all(prey[i]), greater<int>());
    // ff(i, 1, m) ff(j, 1, n) cout << prey[i][j] << " \n"[j == n];
    int minn = -INF;
    ff(i, 1, m)
    {
        int temp = 0;
        ff(j, 1, n / 2 + 1)
        {
            temp += prey[i][j];
        }
        minn = max(minn, temp);
    }
    cout << ans + minn << endl;
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