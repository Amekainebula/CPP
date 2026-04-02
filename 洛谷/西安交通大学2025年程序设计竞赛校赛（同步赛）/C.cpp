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
double ans[1 << 9];
void Murasame()
{
    int n;
    cin >> n;
    ff(i, 0, (1 << n) - 1)
    {
        double a, b;
        cin >> a >> b;
        a = a * a + b * b;
        int k = i, cnt = 0;
        while (k)
        {
            if (k & 1)
            {
                ans[cnt] += a;
            }
            k >>= 1;
            cnt++;
        }
    }
    ff(i, 0, n - 1)
    {
        cout << fixed << setprecision(10) << 1 - ans[i] << " " << ans[i] << endl;
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