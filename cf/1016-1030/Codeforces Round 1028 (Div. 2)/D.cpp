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
const int N = 1e6 + 5;
vi a(N), b(N), c(N), x(N), y(N), z(N);
void Murasame()
{
    int n, q;
    cin >> n >> q;
    ff(i, 1, n)
    {
        cin >> b[i];
        c[i] = b[i];
    }
    ff(i, 1, q)
    {
        cin >> x[i] >> y[i] >> z[i];
    }
    ffg(i, q, 1)
    {
        int temp = c[z[i]];
        c[z[i]] = 0;
        c[x[i]] = max(c[x[i]], temp);
        c[y[i]] = max(c[y[i]], temp);
    }
    ff(i, 1, n)
    {
        a[i] = c[i];
    }
    ff(i, 1, q)
    {
        c[z[i]] = min(c[x[i]], c[y[i]]);
    }
    ff(i, 1, n)
    {
        if (c[i] != b[i])
        {
            cout << "-1" << endl;
            return;
        }
    }
    ff(i, 1, n)
    {
        cout << a[i] << " ";
    }
    cout << endl;
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