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
const int N = 2e5 + 5;
int qpow(int a, int b)
{
    int ans = 1;
    a %= mod;
    while (b)
    {
        if (b & 1)
            ans = (ans * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return ans;
}
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1), b(n + 1);
    ff(i, 1, n) cin >> a[i];
    ff(i, 1, n) cin >> b[i];
    int maxa = -1, maxb = -1;
    int ida = 0, idb = 0;
    int ans;
    ff(i, 1, n)
    {
        if (a[i] > maxa)
        {
            maxa = a[i];
            ida = i;
        }
        if (b[i] > maxb)
        {
            maxb = b[i];
            idb = i;
        }
        if (maxa == maxb)
        {
            ans = qpow(2, maxa) + qpow(2, max(a[i - idb + 1], b[i - ida + 1]));
            cout << ans % mod << " ";
        }
        else
        {
            if (maxa > maxb)
            {
                ans = qpow(2, maxa) + qpow(2, b[i - ida + 1]);
                cout << ans % mod << " ";
            }
            else
            {
                ans = qpow(2, maxb) + qpow(2, a[i - idb + 1]);
                cout << ans % mod << " ";
            }
        }
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