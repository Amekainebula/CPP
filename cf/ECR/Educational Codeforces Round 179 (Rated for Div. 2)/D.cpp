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
void Murasame()
{
    int n, m;
    cin >> n >> m;
    vi a(m + 1);
    ff(i, 1, m)
    {
        cin >> a[i];
    }
    sort(all1(a));
    int cnt = 0;
    ff(i, 1, (m + 1) / 2)
    {
        ff(j, 1, 2)
        {
            ff(k, 1, 3)
            {
                if (j == 1)
                    cout << a[i] << " " << a[m - i + 1] << " ";
                else
                    cout << a[m - i + 1] << " " << a[i] << " ";
            }
            cnt++;
            cout << endl;
            if (cnt == n)
                return;
        }
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
        Murasame();
    }
    return 0;
}